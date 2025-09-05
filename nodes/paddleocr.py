import json
import os
import re
import base64
from io import BytesIO
import io
import requests
from retry import retry
import boto3
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import cast
import numpy as np
import torch
import cv2
import folder_paths
import torch
from torchvision import transforms
from paddleocr import PaddleOCR
import tempfile
import folder_paths

models_dir = folder_paths.models_dir

current_directory = os.path.dirname(os.path.abspath(__file__))
temp_img_path = os.path.join(current_directory, "temp_dir", "AnyText_manual_mask_pos_img.png")
MAX_RETRY = 3


## for layer style nodes
class ImageOCRByPaddleOCR:

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":{
                  "image": ("IMAGE",),
                  }
                }
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","INT", "INT", "STRING", "IMAGE","IMAGE")
    RETURN_NAMES = ("Texts","x_offsets","y_offsets","widths","heights","img_width","img_height", "ocr_results_json", "Mask Image","Result Image")
    FUNCTION = "forward"
    CATEGORY = "paddleocr"
    OUTPUT_NODE = True

    def convert_to_xywh_old(self,coordinates):
        # Extract all x and y coordinates
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]

        # Calculate x offset and y offset
        x_offset = min(x_coords)
        y_offset = min(y_coords)

        # Calculate width and height
        width = max(x_coords) - x_offset
        height = max(y_coords) - y_offset

        print(x_offset,y_offset,width,height)
        return int(x_offset), int(y_offset), int(width), int(height)
    
    def convert_to_xywh(self,coordinates):
        
        # Calculate x offset and y offset
        x_offset = coordinates[0]
        y_offset = coordinates[1]

        # Calculate width and height
        width = coordinates[2] - x_offset
        height = coordinates[3] - y_offset

        #print(x_offset,y_offset,width,height)
        return int(x_offset), int(y_offset), int(width), int(height)

    def ocr_by_paddleocr(self,image_input):

        print('image_input shape is', image_input.shape)

        image = image_input[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        numpy_image = (image_input[0] * 255.0).clamp(0, 255).numpy()

        img_width, img_height = image.size


        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='/tmp/') as temp_file:
            temp_filename = temp_file.name

        # Save numpy_image as a temporary file
        cv2.imwrite(temp_filename, cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR))

        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)

        #ocr_results = ocr.ocr(temp_filename, cls=True)[0]
        ocr_results = ocr.ocr(temp_filename)[0]
        ocr_results.save_to_img("output")
        #print("!!!!!!!!!!!!!ocr_results", ocr_results)
        #print(ocr_results.keys())
        # Create a mask image with the same size as the original image
        result = []
        masked_img = np.array(Image.new("RGB", (numpy_image.shape[1], numpy_image.shape[0]), (0, 0, 0)))
        result_img = numpy_image.copy()
        all_text=""
        x_offsets=[]
        y_offsets=[]
        widths=[]
        heights=[]

        print("ocr_results", ocr_results['rec_boxes'])
        # Extract text and bounding box information
        #for line in ocr_results:
        for index in range(len(ocr_results['rec_texts'])):
             boxes = ocr_results['rec_boxes'][index]
             #print("boxes",boxes)
             x_offset,y_offset,width,height = self.convert_to_xywh(boxes)
             x_offsets.append(str(x_offset))
             y_offsets.append(str(y_offset))
             widths.append(str(width))
             heights.append(str(height))

             #text = line[1][0]
             text = ocr_results['rec_texts'][index]
             #print("text",text)
             result.append(text)

             # Draw mask for each text information box
             # Specify the coordinates of the top-left and bottom-right corners of the rectangle
             x1, y1 = int(x_offset), int(y_offset)
             x2, y2 = int(x_offset + width), int(y_offset + height)
             # Draw a black rectangle on the mask image
             cv2.rectangle(masked_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
             cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 0)


        ocr_results_json = {}
        for index in range(len(result)):
            ocr_results_json[index] = {
                "text": result[index],
                "bbox": [int(x_offsets[index]), int(y_offsets[index]), int(x_offsets[index]) + int(widths[index]), int(y_offsets[index]) + int(heights[index])],
            }

        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)
        result_img = torch.from_numpy(np.array(result_img).astype(np.float32) / 255.0).unsqueeze(0)
        all_text="|".join(result)
        x_offsets="|".join(x_offsets)
        y_offsets="|".join(y_offsets)
        widths="|".join(widths)
        heights="|".join(heights)



        print("result",result)

        # Add original image output
        original_img = image_input
        # delete temp files
        os.unlink(temp_filename)

        return all_text,x_offsets,y_offsets,widths,heights,img_width,img_height, json.dumps(ocr_results_json, ensure_ascii=False), masked_img, result_img

    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_paddleocr(image)

class TextInformationMask:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "image": ("IMAGE", ),
                    "ocrRes": ("STRING", {"multiline": True}),
                }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "mask_img")
    FUNCTION = "TextInformationMask"

    CATEGORY = "postprocessingTool"

    def TextInformationMask(self, image, ocrRes):

        image = image[0] * 255.0
        image = image.clamp(0, 255).numpy().round().astype(np.uint8)
        print("image shape is", image.shape)

        json_start = ocrRes.find('{')
        json_end = ocrRes.rfind('}') + 1
        print(ocrRes[json_start:json_end])
        if json_start >= 0 and json_end > json_start:
            result_json = json.loads(ocrRes[json_start:json_end])
        else:
            raise ValueError("Invalid JSON string")
        result_list = result_json['text_information']

        # Extract text and bounding box information
        mask_img = np.array(Image.new("RGB", (image.shape[1], image.shape[0]), (0, 0, 0)))
        
        #for line in ocr_results:
        for index in range(len(result_list)):
             ocr_result_re = result_list[index]
             print("ocr_result_re is ", ocr_result_re)

             word_list = ocr_result_re.get('word_list', [])
             if len(word_list) > 0:
                for word in word_list:
                    bbox = word.get('bbox', [0, 0, 0, 0])
                    x1, y1, x2, y2 = bbox
                    # black areas will be inpainted
                    cv2.rectangle(mask_img, (x1, y1), (x2, y2), 255, -1)
             else:
                bbox = ocr_result_re.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox

                #add Semi-transparent masks with color red for image in box (x1,y1,x2,y2)
                overlay = image.copy()
                alpha = 0.4  # 透明度 0.0~1.0（越低越透明）

                # 在 overlay 上画实心矩形（厚度 = -1 表示填充）
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), thickness=-1)

                # 将 overlay 合成到原图（在框区域做加权）
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
                # add red box to the original image
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        mask_img = torch.from_numpy(np.array(mask_img).astype(np.float32) / 255.0).unsqueeze(0)
        print("mask image shape is", mask_img.shape)
        print("result image shape is", image.shape)
        return image, mask_img


def is_chinese_char(ch):
    """判断一个字符是否是中文"""
    code_point = ord(ch)
    return (
        0x4E00 <= code_point <= 0x9FFF or
        0x3400 <= code_point <= 0x4DBF or
        0x20000 <= code_point <= 0x2A6DF or
        0x2A700 <= code_point <= 0x2B73F or
        0x2B740 <= code_point <= 0x2B81F or
        0x2B820 <= code_point <= 0x2CEAF or
        0x3000 <= code_point <= 0x303F
    )
def wrap_text(text, font, max_width):
    """文本自动换行"""
    bbox = font.getbbox(text)
    if bbox[2] - bbox[0] <= max_width:
        return [text]
    
    lines = []
    if is_chinese_char(text[0]):
        words = list(text)
    else:
        words = text.split()
    
    line = ""
    for word in words:
        test_line = line + word
        if not is_chinese_char(text[0]):
            test_line += ' '
        
        bbox_test = font.getbbox(test_line)
        if bbox_test[2] - bbox_test[0] <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line.strip())
            line = word + (' ' if not is_chinese_char(text[0]) else '')
    
    if line:
        lines.append(line.strip())
    
    return lines
class TextOverLay:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": 
                {
                    "backgroud_image": ("IMAGE", ),
                    "text_information": ("STRING", {"multiline": True}),
                }
        }

    RETURN_TYPES = ("IMAGE","STRING")
    RETURN_NAMES = ("image", "text_information_json")
    FUNCTION = 'text_overlay'
    CATEGORY = 'postprocessingTool'

    @staticmethod
    def collect_font_files(font_folder_path:str, suffixes:tuple):
        result = {}
        for filename in os.listdir(font_folder_path):
            true_ext = os.path.splitext(filename)[1]
            if true_ext.lower() not in suffixes:
                continue
            full_path = os.path.join(font_folder_path, filename)
            result.update({filename: full_path})
        return result
    
    @staticmethod
    def get_max_fontsize(text, font_path, box_width, box_height, max_size=300):
        """
        Find the largest font size that fits the text within the given box.
        """
        for font_size in range(1, max_size + 1):
            font = ImageFont.truetype(font_path, font_size)
            bbox = font.getbbox(text)  # (left, top, right, bottom)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            if box_height >0:
                if text_height > box_height:
                    return font_size - 1  # last acceptable size
            elif box_width > 0:
                if text_width > box_width:
                    return font_size - 1  # last acceptable size
            else:
                raise ValueError("Be sure one of box_width and box_height larger than zero.")

        return max_size  # text fits even at max_size

    def text_overlay(self, backgroud_image, text_information):

        image = backgroud_image[0] * 255.0
        json_start = text_information.find('{')
        json_end = text_information.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            text_information = json.loads(text_information[json_start:json_end])
        else:
            raise ValueError("Invalid JSON string")
    
        text_list = text_information.get('text_information', [])
        if text_list is None or len(text_list) == 0:
            return backgroud_image, json.dumps(text_information, ensure_ascii=False)
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        current_directory = os.path.dirname(os.path.abspath(__file__))
        font_resource_dir = os.path.join(current_directory, "font_dir")
        print("font_resource_dir is", font_resource_dir)
        FONT_DICT = {}
        FONT_DICT.update(TextOverLay.collect_font_files(font_resource_dir, ('.ttf', '.otf'))) # 后缀要小写
        print('All aviable font files is: ', FONT_DICT)
        font_file = text_information.get('font_file', 'NotoSansSC.ttf')
        regular_font_path = FONT_DICT.get(font_file)
        # Check if the font path exists and set to default if not provided
        if not regular_font_path:
            regular_font_path = os.path.join(font_resource_dir, 'NotoSansSC.ttf')
            print(f"Font file '{font_file}' not found in the font directory. Using default font: {regular_font_path}")

        draw = ImageDraw.Draw(image)

        line_break = True
        
        item_index = 0
        for item in text_list:
            text = item.get('text', '').strip()
            if not text:
                continue
                
            bbox = item.get('bbox', [0, 0, 100, 30])
            x1, y1, x2, y2 = bbox
            text_width = x2 - x1
            text_height = y2 - y1
            
            
            text_color = item.get('text_color', '#FFFFFF')
            alignment = item.get('alignment', 'center')
            if text_width <=0:
                alignment = 'left'

            leading = item.get('leading', 8)

            bold = item.get('bold', "false")
            if bold == "true":
                font_path = regular_font_path.replace('.ttf', '-Bold.ttf')
            else:
                font_path = regular_font_path
            
            
            paragraphs = text.split('\n')
            row_number = len(paragraphs)

            if row_number > 1:
                line_break = False

            font_size = item.get('font_size', 0)
            if font_size <= 0:
                if text_width >0 or text_height >0:
                    font_size = self.get_max_fontsize(text, font_path, text_width, float(text_height)/row_number)
                else:
                    font_size = 30
            text_information["text_information"][item_index]["font_size"] = font_size
            item_index += 1

            font = ImageFont.truetype(font_path, font_size)

            single_row_height = int(text_height/row_number)
            
            y_offset = y1
            print('!!!!!!!!!', paragraphs, y_offset, font_path)
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                
                if line_break:
                    lines = wrap_text(paragraph, font, text_width)
                else:
                    lines = [paragraph]
                
                for line in lines:
                    bbox_line = font.getbbox(line)
                    line_width = bbox_line[2] - bbox_line[0]
                    
                    if alignment == "left":
                        x_text = x1
                    elif alignment == "right":
                        x_text = x2 - line_width
                    else:  # center
                        x_text = x1 + (text_width - line_width) // 2
                    
                    tw, th = bbox_line[2] - bbox_line[0], bbox_line[3] - bbox_line[1]
                    y_corrected = y_offset + (single_row_height - th)/2 - bbox_line[1]
                    
                    draw.text(
                        xy=(x_text, y_corrected),
                        text=line,
                        fill=text_color,
                        font=font
                    )
                    
                    y_offset += (bbox_line[3] - bbox_line[1]) + leading
        
        result_img = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        print("result image shape is", result_img.shape)
        return result_img, json.dumps(text_information, ensure_ascii=False)

class SaveText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "file_name": ("STRING", {"default": 'result.txt', "multiline": False}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_text"
    OUTPUT_NODE = True
    CATEGORY = "text"

    def save_text(self, text, file_name):
        output_dir = folder_paths.get_output_directory()
        save_path = os.path.join(output_dir, file_name)
        #get the suffix of the file_name
        suffix = os.path.splitext(file_name)[1].lower()
        results = []
        if suffix == '.txt':
            # Write the text to the file
            with open(save_path, 'w', encoding='utf-8') as file:
                file.write(text)
            
        
        if suffix == '.json':
            # Write the text to a json file
            import json
            with open(save_path, 'w', encoding='utf-8') as file:
                json.dump(text, file, ensure_ascii=False, indent=4)
        result={"text": text, "filename": file_name, "type": "output"}
        results.append(result)

        return {"ui": {"text": results}}

NODE_CLASS_MAPPINGS = {
    "Image OCR by PaddleOCR": ImageOCRByPaddleOCR,
    "Text Information Mask": TextInformationMask,
    "Text Overlay": TextOverLay,
    "Save Text": SaveText
}

