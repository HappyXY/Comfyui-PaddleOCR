import json
import os
import re
import base64
from io import BytesIO
import io
import requests
from retry import retry
import boto3
from PIL import Image
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

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING","IMAGE","IMAGE")
    RETURN_NAMES = ("Texts","x_offsets","y_offsets","widths","heights","img_width","img_height","Mask Image","Original Image")
    FUNCTION = "forward"
    CATEGORY = "aws"
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

        print(x_offset,y_offset,width,height)
        return int(x_offset), int(y_offset), int(width), int(height)

    def ocr_by_paddleocr(self,image_input):

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
             print("boxes",boxes)
             x_offset,y_offset,width,height = self.convert_to_xywh(boxes)
             x_offsets.append(str(x_offset))
             y_offsets.append(str(y_offset))
             widths.append(str(width))
             heights.append(str(height))

             #text = line[1][0]
             text = ocr_results['rec_texts'][index]
             print("text",text)
             result.append(text)

             # Draw mask for each text information box
             # Specify the coordinates of the top-left and bottom-right corners of the rectangle
             x1, y1 = int(x_offset), int(y_offset)
             x2, y2 = int(x_offset + width), int(y_offset + height)
             # Draw a black rectangle on the mask image
             cv2.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        masked_img = torch.from_numpy(np.array(masked_img).astype(np.float32) / 255.0).unsqueeze(0)

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

        return all_text ,x_offsets,y_offsets,widths,heights,img_width,img_height, masked_img,original_img



    @retry(tries=MAX_RETRY)
    def forward(self, image):
        return self.ocr_by_paddleocr(image)



NODE_CLASS_MAPPINGS = {
    "Image OCR by PaddleOCR": ImageOCRByPaddleOCR
}

