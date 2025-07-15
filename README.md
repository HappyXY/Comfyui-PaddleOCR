# ComfyUI-PaddleOCR

A comprehensive ComfyUI custom node package for OCR (Optical Character Recognition) and advanced text rendering capabilities using PaddleOCR.

## Features

### 🔍 OCR Capabilities
- **Text Recognition**: Extract text from images with high accuracy using PaddleOCR
- **Layout Analysis**: Analyze document structure and text positioning
- **Bounding Box Detection**: Get precise coordinates of detected text regions
- **Multi-language Support**: Supports various languages through PaddleOCR
- **JSON Output**: Structured OCR results with text content and positioning data

### 🎨 Text Rendering & Overlay
- **JSON-based Text Configuration**: Define text properties through JSON input including:
  - Text content and positioning
  - Font selection from included font library
  - Font size and styling options
  - Text color and stroke effects
  - Text alignment and wrapping
- **Rich Font Library**: Includes multiple Chinese fonts (华文系列, 微软雅黑)
- **Advanced Typography**: Support for character-per-line control, leading, and text wrapping
- **Stroke Effects**: Customizable text outlines with color and width control

## Available Nodes

### 1. Image OCR by PaddleOCR
Performs OCR on input images and extracts text with positioning information.

**Inputs:**
- `image`: Input image for OCR processing

**Outputs:**
- `Texts`: Extracted text content
- `x_offsets`, `y_offsets`: Text position coordinates
- `widths`, `heights`: Text bounding box dimensions
- `img_width`, `img_height`: Original image dimensions
- `ocr_results_json`: Complete OCR results in JSON format
- `Mask Image`: Generated mask for detected text regions
- `Result Image`: Annotated image with OCR results

### 2. Text Image Overlay
Renders text onto images with extensive customization options.

**Inputs:**
- `background_image`: Base image for text overlay
- `text`: Text content to render (supports multiline)
- `font_file`: Font selection from available fonts
- `align`: Text alignment options
- `char_per_line`: Characters per line for text wrapping
- `leading`: Line spacing control
- `font_size`: Text size (1-2500px)
- `text_color`: Text color specification
- `stroke_width`: Outline thickness
- `stroke_color`: Outline color
- `x_offset`, `y_offset`: Text positioning
- `text_width`, `text_height`: Text area dimensions

### 3. BBOX to Mask
Converts bounding box coordinates to image masks.

### 4. Copy Translation Postprocess
Processes translation results and applies them to images.

## JSON Configuration Format

The text rendering system supports JSON input for complex text layouts:

```json
{
  "text": "Your text content",
  "font": "font_name.ttf",
  "size": 72,
  "color": "#FFFFFF",
  "stroke_color": "#000000",
  "stroke_width": 2,
  "align": "center",
  "x": 100,
  "y": 200,
  "width": 400,
  "height": 100
}
```

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/Comfyui-PaddleOCR.git
```

2. Install required dependencies:
```bash
pip install paddleocr pillow opencv-python numpy torch torchvision
```

3. Restart ComfyUI to load the new nodes.

## Font Management

The package includes a comprehensive font library located in `nodes/font_dir/`:
- Multiple Chinese fonts (华文中宋, 华文仿宋, 华文彩云, etc.)
- Microsoft YaHei (微软雅黑) regular and bold
- Support for TTF and OTF font formats

To add custom fonts:
1. Place font files (.ttf or .otf) in the `nodes/font_dir/` directory
2. Restart ComfyUI to refresh the font list

## Use Cases

- **Document Processing**: Extract and analyze text from scanned documents
- **Image Translation**: OCR text extraction followed by translation overlay
- **Content Creation**: Add styled text overlays to images
- **Data Extraction**: Automated text extraction from images for data processing
- **Multilingual Content**: Handle text in various languages with appropriate fonts

## Requirements

- ComfyUI
- Python 3.8+
- PaddleOCR
- PIL (Pillow)
- OpenCV
- NumPy
- PyTorch

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
