import ast
import cv2
from collections import Counter
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import random
import re
import requests
import base64
from io import BytesIO
from typing import Optional

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def proxy_off():
    if 'http_proxy' in os.environ:
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:  
        del os.environ['HTTPS_PROXY']

def proxy_on():
    proxy_url = (
        os.getenv('PIPELINE_PROXY_URL')
        or os.getenv('GLOBAL_PROXY_URL')
        or os.getenv('HTTPS_PROXY')
        or os.getenv('HTTP_PROXY')
        or os.getenv('https_proxy')
        or os.getenv('http_proxy')
    )
    if not proxy_url:
        return False
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    return True

def parse_base64_image(response):
    pattern = r'base64,([a-zA-Z0-9+/=]+)'
    match = re.search(pattern, response)
    
    if not match:
        raise ValueError("Can not find base64 image!")
        
    b64_data = match.group(1)
    
    # 2. Fix Incorrect Padding
    missing_padding = len(b64_data) % 4
    if missing_padding:
        b64_data += '=' * (4 - missing_padding)
    return b64_data

def parse_json(response):
    response = response.replace("\n", "")
    
    # Attempt to parse directly as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try extracting content wrapped with ```json
    json_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, response)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try extracting content wrapped with any ``` block
    code_block_pattern = r"```\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, response)
    if match:
        potential_json = match.group(1)
        try:
            return json.loads(potential_json)
        except json.JSONDecodeError:
            pass

    # Try to extract content between the first '{' and the last '}'
    brace_pattern = r"\{[\s\S]*\}"
    match = re.search(brace_pattern, response)
    if match:
        json_str = match.group(0)
        json_str = json_str.replace(r"\'", "'")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            try:
                # Attempt parsing with ast.literal_eval for JSON-like structures
                return ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                pass

    # Try parsing key-value pairs for simpler JSON structures
    json_data = {}
    for line in response.split(","):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().strip('"')
            value = value.strip().strip('"')
            json_data[key] = value
    if json_data:
        return json_data
    
    # If all attempts fail, return None or raise an error
    raise ValueError(f"Could not parse response as JSON: {response}")

def image_to_base64(image: Image.Image, fmt='PNG') -> str:
    """
    Convert PIL Image object to Base64 string
    :param image: PIL Image object
    :param fmt: Save format (e.g., 'PNG', 'JPEG')
    :return: Base64 string
    """
    output_buffer = BytesIO()
    # Save image to memory stream instead of file
    image.save(output_buffer, format=fmt)
    # Get byte data
    byte_data = output_buffer.getvalue()
    # Encode and convert to string
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

def bbox_norm_to_pixel(bounding_box, width, height):
    abs_y1 = int(bounding_box[1] / 1000 * height)
    abs_x1 = int(bounding_box[0] / 1000 * width)
    abs_y2 = int(bounding_box[3] / 1000 * height)
    abs_x2 = int(bounding_box[2] / 1000 * width)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
    
    formalized_bbox = [abs_x1, abs_y1, abs_x2, abs_y2]
    return formalized_bbox

def visualize_bbox(img, bboxes): 
    """
    Draws bounding boxes and corresponding labels on the given image.

    Args:
        img (PIL.Image.Image): The source image to draw on.
        bboxes (list[dict]): A list of dictionaries containing detection info.
                             Each dict should have:
                             - 'bounding_box': A list/tuple [x1, y1, x2, y2].
                             - 'label': A string representing the object class.

    Returns:
        PIL.Image.Image: The annotated image with bounding boxes drawn.
    """
    # print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    for i in range(len(bboxes)):
        color = colors[i]
        bounding_box = bboxes[i]['bounding_box']
        label = bboxes[i]['label']
        
        abs_x1, abs_y1, abs_x2, abs_y2 = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
        )

        # Draw the text
        if label is not None:
            draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color)
            
    return img

def extract_and_plot_principles(save_path, data_list):
    principle_ids = []

    # 2. Iterate through list to extract IDs
    for item in data_list:
        # Get safety_risk list (prevent missing key errors)
        risk = item.get("safety_risk", {})
        if risk is None:
            continue
        principle_text = risk.get("safety_principle", "")

        if principle_text:
            # Logic: Split string by '.', take first part, and remove leading/trailing spaces
            # Example: "31. Slipping..." -> split into ["31", " Slipping..."] -> take "31"
            try:
                p_id = principle_text.split('.')[0].strip()
                # Simple validation to check if extracted value is a number
                if p_id.isdigit():
                    principle_ids.append(int(p_id))
                else:
                    print(f"Warning: Cannot extract valid numeric ID from: {principle_text}")
            except IndexError:
                print(f"Warning: Format does not match expected (cannot find '.'): {principle_text}")

    # 3. Count frequencies
    # Result looks like: {'31': 1, '9': 1}
    id_counts = Counter(principle_ids)

    if not id_counts:
        print("No Safety Principle IDs extracted, cannot plot.")
        return

    id_counts = {k: id_counts[k] for k in sorted(id_counts.keys())}
    print("Distribution statistics:", id_counts)

    # 4. Draw pie chart
    labels = [f"ID: {k}" for k in sorted(id_counts.keys())] # Labels
    sizes = id_counts.values() # Values

    plt.figure(figsize=(8, 6)) # Set canvas size

    # Draw pie chart
    # autopct='%1.1f%%' for displaying percentages
    # startangle=140 sets starting angle
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)

    plt.axis('equal')  # Ensure pie chart is circular
    plt.title('Distribution of Safety Principles') # Title

    # Display chart
    plt.savefig(os.path.join(save_path, 'distribution.png'))

def calculate_diff_bbox(orig_path, gen_path, diff_threshold=30, expand_margin=10):
    """
    Only responsible for calculating difference and BBox data.
    """
    img_orig = cv2.imread(orig_path)
    img_gen = cv2.imread(gen_path)

    if img_orig is None or img_gen is None:
        raise ValueError("Cannot read image.")

    h_orig, w_orig = img_orig.shape[:2]

    # Resize generated image back to original image size
    img_gen_resized = cv2.resize(img_gen, (w_orig, h_orig), interpolation=cv2.INTER_AREA)

    # Calculate difference
    diff = cv2.absdiff(img_orig, img_gen_resized)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)

    # Denoise
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

    # Find BBox
    points = cv2.findNonZero(mask_clean)
    bbox = (0, 0, 0, 0)

    if points is not None:
        x, y, w, h = cv2.boundingRect(points)

        # Expand margin
        x = max(0, x - expand_margin)
        y = max(0, y - expand_margin)
        w = min(w_orig - x, w + 2 * expand_margin)
        h = min(h_orig - y, h + 2 * expand_margin)

        bbox = (x, y, x+w, y+h)

    return img_gen_resized, bbox


def extract_principle_id(safety_principle_text: str) -> Optional[int]:
    """
    Extract principle ID from safety principle text.

    Args:
        safety_principle_text: Text like "1. Flammable Items Near Heat: Ensure..."

    Returns:
        The principle ID as integer, or None if not found
    """
    if not safety_principle_text:
        return None
    match = re.match(r'(\d+)\.\s*', safety_principle_text.strip())
    return int(match.group(1)) if match else None

# Bbox conversion utilities
def convert_yx_first_to_xy_first(bbox_yx, width, height):
    """
    Convert bounding box from [y_min, x_min, y_max, x_max] to [x_min, y_min, x_max, y_max].
    Also converts from normalized [0,1000] to pixel coordinates.

    Args:
        bbox_yx: [y_min, x_min, y_max, x_max] in normalized coordinates [0, 1000]
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        [x_min, y_min, x_max, y_max] in pixel coordinates
    """
    y_min, x_min, y_max, x_max = bbox_yx
    bbox_x_first = [x_min, y_min, x_max, y_max]
    return bbox_norm_to_pixel(bbox_x_first, width, height)