from PIL import ImageFont, ImageDraw, Image,ImageChops
import numpy as np
import os
import cv2
import random
from WaterMark import get_pngs_with_transparent_background
def crop_transparent_padding(png_path, alpha_threshold=20):
    img = Image.open(png_path).convert("RGBA")
    alpha = img.getchannel("A")
    
    # Create a binary mask where alpha >= threshold
    binary_alpha = alpha.point(lambda p: 255 if p >= alpha_threshold else 0)
    
    # Get bounding box of non-transparent areas
    bbox = binary_alpha.getbbox()
    
    return img.crop(bbox) if bbox else img

def add_red_background(img):
    red_bg = Image.new("RGBA", img.size, (255, 0, 0, 255))
    red_bg.paste(img, (0, 0), img)
    return red_bg

# Paths
logo_dir = r"./Logos"
output_dir = r"C:\Users\Altuner\Desktop\Test"
os.makedirs(output_dir, exist_ok=True)

# Get logos with transparent background
logo_files = get_pngs_with_transparent_background(logo_dir)

# Apply to first 30
for i, logo_name in enumerate(logo_files[:300]):
    input_path = os.path.join(logo_dir, logo_name)
    output_path = os.path.join(output_dir, f"cropped_{i+1}.png")

    try:
        cropped = crop_transparent_padding(input_path)
        with_bg = add_red_background(cropped)
        with_bg.save(output_path)
    except Exception as e:
        print(f"Error processing {logo_name}: {e}")