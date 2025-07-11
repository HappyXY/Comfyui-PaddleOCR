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
    
    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING", "STRING", "STRING", "IMAGE","IMAGE")
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

        print('!!!!!!!!!!!image_input', image_input.shape)

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
        masked_img = numpy_image.copy()
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
             cv2.rectangle(masked_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
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


class BBOXTOMASK:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "bboxs": ("LIST", ),
                    "width": ("INT", ),
                    "height": ("INT", )
                }
                }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASK",)
    FUNCTION = "bbox2mask"

    CATEGORY = "preprocessingTool"

    def bbox2mask(self, bboxs, width, height):
        masked_img = Image.new("L", (width, height), 0)
        for bbox in bboxs:
            x1, y1, x2, y2 = bbox
            # Draw a white rectangle on the mask image
            masked_img.paste(255, (x1, y1, x2, y2)) 
        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        #print('!!!!!!!! mask shape', masked_img.shape)
        return masked_img

class CopyTransPostprocess:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "image": ("IMAGE", ),
                    "copyTransRes": ("STRING", {"multiline": True}),
                }
                }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "LIST", "STRING", "IMAGE")
    RETURN_NAMES = ("Texts","x_offsets","y_offsets","widths","heights", "bboxes", "text_colors", "image")
    FUNCTION = "CopyTranslationPostprocess"

    CATEGORY = "postprocessingTool"

    def CopyTranslationPostprocess(self, image, copyTransRes):
        json_start = copyTransRes.find('{')
        json_end = copyTransRes.rfind('}') + 1
        print(copyTransRes[json_start:json_end])
        if json_start >= 0 and json_end > json_start:
            result_json = json.loads(copyTransRes[json_start:json_end])
        result_list = result_json['copytext_translation_results']
        
        result = []
        x_offsets=[]
        y_offsets=[]
        widths=[]
        heights=[]
        text_colors=[]

        all_boxes = []
        # Extract text and bounding box information
        #for line in ocr_results:
        for index in range(len(result_list)):
             copy_trans_re = result_list[index]
             print("copy_trans_re",copy_trans_re)
             bbox = copy_trans_re.get('bbox', [0, 0, 0, 0])
             all_boxes.append(bbox)
             x1, y1, x2, y2 = bbox
             x_offsets.append(str(x1))
             y_offsets.append(str(y1))
             widths.append(str(x2-x1))
             heights.append(str(y2-y1))

             #text = line[1][0]
             text = copy_trans_re.get('chinese_translation', '')
             print("text",text)
             result.append(text)

             text_color = copy_trans_re.get('text_color', '#FFFFFF')
             text_colors.append(text_color)
             cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 0)
        
        all_text=";".join(result)
        x_offsets=";".join(x_offsets)
        y_offsets=";".join(y_offsets)
        widths=";".join(widths)
        heights=";".join(heights)
        text_colors=";".join(text_colors)
        image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        return all_text, x_offsets, y_offsets, widths, heights, all_boxes, text_colors, image


class TextImageOverLay:

    @classmethod
    def INPUT_TYPES(cls):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        font_resource_dir = os.path.join(current_directory, "font_dir")
        FONT_DICT = {}
        FONT_DICT.update(TextImageOverLay.collect_font_files(font_resource_dir, ('.ttf', '.otf'))) # 后缀要小写
        FONT_LIST = list(FONT_DICT.keys())
        return {
            "required": {
                "background_image": ("IMAGE",),
                "text": ("STRING",{"default": "text", "multiline": True},
                ),
                "font_file": (FONT_LIST,),
                "align": (["center", "left", "right"],),
                "char_per_line": ("INT", {"default": 80, "min": 1, "max": 8096, "step": 1},),
                "leading": ("INT",{"default": 8, "min": 0, "max": 8096, "step": 1},),
                "font_size": ("INT",{"default": 72, "min": 1, "max": 2500, "step": 1},),
                "text_color": ("STRING", {"multiline": True},),
                "stroke_width": ("INT",{"default": 0, "min": 0, "max": 8096, "step": 1},),
                "stroke_color": ("STRING",{"default": "#FF8000"},),
                "x_offset": ("STRING", {"multiline": True},),
                "y_offset": ("STRING", {"multiline": True},),
                "text_width": ("STRING", {"multiline": True},),
                "text_height": ("STRING", {"multiline": True},),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'text_image_overlay'
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
            
            if text_width > box_width or text_height > box_height:
                return font_size - 1  # last acceptable size

        return max_size  # text fits even at max_size

    def text_image_overlay(self, background_image, text, font_file, align, char_per_line,
                          leading, font_size, text_color,
                          stroke_width, stroke_color, x_offset, y_offset,
                          text_width, text_height
                          ):

        image = background_image[0] * 255.0
        image = Image.fromarray(image.clamp(0, 255).numpy().round().astype(np.uint8))
        current_directory = os.path.dirname(os.path.abspath(__file__))
        font_resource_dir = os.path.join(current_directory, "font_dir")
        print("!!!!!!!!!! font_resource_dir", font_resource_dir)
        FONT_DICT = {}
        FONT_DICT.update(TextImageOverLay.collect_font_files(font_resource_dir, ('.ttf', '.otf'))) # 后缀要小写
        width, height = image.size
        font_path = FONT_DICT.get(font_file)
        text_list = re.split('[,;]+', text)
        x_offset_list = re.split('[,;]+', x_offset)
        y_offset_list = re.split('[,;]+', y_offset)
        text_width_list = re.split('[,;]+', text_width)
        text_height_list = re.split('[,;]+', text_height)
        text_color_list = re.split('[,;]+', text_color)

        print("!!!!!!!!!! text_list", text_list)
        print("!!!!!!!!!! x_offset_list", x_offset_list)        
        print("!!!!!!!!!! y_offset_list", y_offset_list)
        print("!!!!!!!!!! text_width_list", text_width_list)
        print("!!!!!!!!!! text_height_list", text_height_list)

        if len(text_list) != len(x_offset_list) or len(text_list) != len(y_offset_list) or \
           len(text_list) != len(text_width_list) or len(text_list) != len(text_height_list):
            raise ValueError("The number of text, x_offsets, y_offset, text_widths, and text_heights must be the same.")
        if not font_path:
            raise ValueError(f"Font file '{font_file}' not found in the font directory.")
        if not os.path.exists(font_path):
            raise ValueError(f"Font file '{font_file}' does not exist at path: {font_path}")
        if not text:
            raise ValueError("Text cannot be empty.")
        image.save('before_draw.png')
        for single_text, x_offset_str, y_offset_str, text_width_str, text_height_str, text_color_str in zip(text_list, x_offset_list, y_offset_list, text_width_list, text_height_list, text_color_list):
            if not single_text.strip():
                continue
            paragraphs = single_text.split('\n')
            x_offset = int(x_offset_str.strip()) if x_offset_str.strip() else 0
            y_offset = int(y_offset_str.strip()) if y_offset_str.strip() else 0
            text_width = int(text_width_str.strip()) if text_width_str.strip() else 0
            text_height = int(text_height_str.strip()) if text_height_str.strip() else 0
            text_height_single_paragraph = text_height / len(paragraphs)
            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                #computer the x and y offsets, text width and height for each paragraph
                x_offset_i = x_offset
                y_offset_i = y_offset + i * text_height_single_paragraph
                text_width_i = text_width
                text_height_i = text_height_single_paragraph
                font_computed_size = TextImageOverLay.get_max_fontsize(paragraph, font_path, text_width_i, text_height_i, max_size=font_size)
                print(f"Computed font size for paragraph '{paragraph}': {font_computed_size}")
                font = cast(ImageFont.FreeTypeFont, ImageFont.truetype(font_path, font_computed_size))
                draw = ImageDraw.Draw(image)
                # 计算文本框的宽度和高度
                bbox = font.getbbox(paragraph)  # (left, top, right, bottom)

                # 根据 align 参数重新计算 x 坐标
                if align == "left":
                    x_text = x_offset_i
                elif align == "center":
                    x_text = int(x_offset_i + (text_width_i / 2) - (bbox[2]-bbox[0])/2)
                elif align == "right":
                    x_text = int(x_offset_i + text_width_i - (bbox[2]-bbox[0]))
                else:
                    x_text = int(x_offset_i + (text_width_i / 2) - (bbox[2]-bbox[0])/2)  # 默认为center对齐
                y_text = y_offset_i
                draw.text(
                    xy=(x_text, y_text),
                    text=paragraph,
                    fill=text_color_str,
                    font=font,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color,
                    )
                image.save(f'after_draw_{single_text}_{i}.png')
        result_img = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        #print("!!!!!!!!!! result_img shape", result_img.shape)
        return result_img

NODE_CLASS_MAPPINGS = {
    "Image OCR by PaddleOCR": ImageOCRByPaddleOCR,
    "BBOX to Mask": BBOXTOMASK,
    "Copy Translation Postprocess": CopyTransPostprocess,
    "Text Image Overlay": TextImageOverLay
}

