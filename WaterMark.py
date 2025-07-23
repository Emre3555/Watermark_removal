
from PIL import ImageFont, ImageDraw, Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import random
import unicodedata
import os
import json
from math import sin, cos, radians,sqrt,tan


FONT_PATH = "./msyh.ttc"
INPUT_DIR = "./CarPhotos"
OUTPUT_DIR = "./Output_Photos"
LOGO_DIR = "./Logos"
THIS_DIR = "./"

def rotate_point_funct(pt, center, angle_deg):
    angle_rad = np.radians(-angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return (rotation_matrix @ (pt - center)) + center

def rotate_image_opencv(pil_img, angle):
    # Convert PIL to NumPy array (RGBA)
    img_np = np.array(pil_img)
    
    # Split RGBA channels
    if img_np.shape[2] == 4:
        b, g, r, a = cv2.split(img_np)
        img_rgb = cv2.merge([b, g, r])
    else:
        raise ValueError("Image must have alpha channel")

    # Get rotation matrix
    h, w = img_rgb.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust matrix for translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    # Rotate RGB and alpha separately
    rotated_rgb = cv2.warpAffine(img_rgb, rot_mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    rotated_a = cv2.warpAffine(a, rot_mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Merge back RGBA
    rotated_rgba = cv2.merge([rotated_rgb[:,:,0], rotated_rgb[:,:,1], rotated_rgb[:,:,2], rotated_a])

    # Convert back to PIL Image
    return Image.fromarray(rotated_rgba, 'RGBA')

def remove_premultiplied_alpha(img: Image.Image) -> Image.Image:
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    arr = np.array(img).astype(np.float32)
    alpha = arr[:, :, 3] / 255.0

    # Avoid division by zero
    nonzero = alpha > 0
    for c in range(3):  # R, G, B
        arr[:, :, c][nonzero] /= alpha[nonzero]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')

def convert_to_grayscale(img: Image.Image, save_path=None):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    r, g, b, a = img.split()
    # Convert RGB to grayscale manually
    # Formula: Gray = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601)
    gray = Image.eval(r, lambda px: int(px * 0.299))
    gray = ImageChops.add(gray, Image.eval(g, lambda px: int(px * 0.587)))
    gray = ImageChops.add(gray, Image.eval(b, lambda px: int(px * 0.114)))
    # Create grayscale RGBA image with original alpha
    gray_rgba = Image.merge("RGBA", (gray, gray, gray, a))
    if save_path:
        gray_rgba.save(save_path)
    return gray_rgba

#Following two functions are the same but one of them takes path as input and the other is image
def crop_alpha(img):
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    return img.crop(bbox) if bbox else img
def crop_transparent_padding(png_path):
    img = Image.open(png_path).convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.getbbox()
    return img.crop(bbox) if bbox else img


def remove_premultiplied_alpha(logo_img: Image.Image) -> Image.Image:
    """
    Removes premultiplied alpha to avoid black halos around transparent logos.
    """
    if logo_img.mode != 'RGBA':
        logo_img = logo_img.convert('RGBA')

    arr = np.array(logo_img).astype(np.float32)
    alpha = arr[:, :, 3] / 255.0

    # Avoid division by zero
    nonzero_alpha = alpha > 0

    for c in range(3):  # R, G, B channels
        arr[:, :, c][nonzero_alpha] /= alpha[nonzero_alpha]

    # Clip and convert back to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')



#for transparent backgrounds
def compare_images(original_path, processed_path, diff_threshold, success_threshold):###diff threshold and success_threshold can be changed
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(processed_path)

    if img1 is None or img2 is None:
        return "Error", None, 1.0

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    diff_pixels = np.count_nonzero(thresh)
    total_pixels = thresh.size
    diff_ratio = diff_pixels / total_pixels

    diff_img = img2.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(diff_img, contours, -1, (0, 0, 255), 2)

    result = True if diff_ratio < success_threshold else False

    return result, diff_ratio

def get_pngs_with_transparent_background(folder_path, threshold_ratio=0.1):
    """
    Scans all PNG files in a folder and returns a list of filenames
    that have transparent backgrounds (based on alpha channel).
    
    Parameters:
        folder_path (str): Path to the folder.
        threshold_ratio (float): Minimum ratio of fully transparent pixels to count as "transparent".
        
    Returns:
        List[str]: Filenames of PNGs with transparent background.
    """
    transparent_files = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            full_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(full_path).convert("RGBA")
                alpha = np.array(img)[:, :, 3]
                total_pixels = alpha.size
                transparent_pixels = np.sum(alpha == 0)
                ratio = transparent_pixels / total_pixels
                if ratio > threshold_ratio:
                    transparent_files.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    return transparent_files




def get_safe_rotated_position(width, height, angle_deg, image_w, image_h):
    """
    Rotated kutunun güvenli yerleştirme aralığını döndürür.
    
    Args:
        width: metin veya logonun birleşik kutusunun genişliği
        height: metin veya logonun birleşik kutusunun yüksekliği
        angle_deg: döndürme açısı (derece)
        image_w: resmin genişliği
        image_h: resmin yüksekliği
    
    Returns:
        ((safe_x_min, safe_x_max), (safe_y_min, safe_y_max))
        Yerleştirme yapılabilecek güvenli x ve y aralıkları
    """

    theta = radians(angle_deg)
    
    # Orijinal kutunun köşeleri
    corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    # Rotasyon matrisi
    R = np.array([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ])
    
    # Rotasyon uygulanmış köşeler
    rotated_corners = np.dot(corners, R.T)
    
    # Döndürülmüş kutunun dış sınırları
    min_x, min_y = np.min(rotated_corners, axis=0)
    max_x, max_y = np.max(rotated_corners, axis=0)

    rotated_w = max_x - min_x
    rotated_h = max_y - min_y

    # Güvenli yerleştirme sınırları
    safe_x_min = int(np.clip(-min_x, 0, image_w - rotated_w))
    safe_x_max = int(np.clip(image_w - max_x, safe_x_min, image_w - rotated_w))
    safe_y_min = int(np.clip(-min_y, 0, image_h - rotated_h))
    safe_y_max = int(np.clip(image_h - max_y, safe_y_min, image_h - rotated_h))

    if safe_x_max <= safe_x_min or safe_y_max <= safe_y_min:
        # Eğer sığmıyorsa ortala
        x_center = image_w // 2 - int(rotated_w // 2)
        y_center = image_h // 2 - int(rotated_h // 2)
        return ((x_center, x_center), (y_center, y_center))

    return ((safe_x_min, safe_x_max), (safe_y_min, safe_y_max))


def is_displayable(char):#helper function
    cat = unicodedata.category(char)
    return cat.startswith('L') or cat.startswith('N') or cat.startswith('P') or cat.startswith('S')


def safe_char_from_range(rng):#helper function
    while True:
        code_point = random.randint(rng[0], rng[1])
        ch = chr(code_point)
        if is_displayable(ch):
            return ch

def put_unicode_text(image, text, position, font_path, font_size, color, angle, opacity):
    # Convert OpenCV BGR image to RGBA PIL Image (add alpha channel)
    font_path = FONT_PATH
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Create a transparent image for the text
    txt_img = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_img)
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text on the transparent image
    draw.text(position, text, font=font, fill=color + (int(255 * opacity),))  # add alpha channel for opacity

    # Rotate the text image around the text position
    # To rotate about the text's position correctly, we shift the origin:
    # Create a mask with the text only, rotated
    rotated = txt_img.rotate(angle, resample=Image.BICUBIC, center=position)

    # Composite the rotated text onto the original image
    image_pil = Image.alpha_composite(image_pil, rotated)

    # Convert back to OpenCV BGR
    return cv2.cvtColor(np.array(image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def applyWaterMark(input_path, output_path, content_type, font, location, pattern, appearance, size, angle, color,logo_files,gray_scale):
    image = cv2.imread(input_path)
    h, w = image.shape[:2]
    text = ""
    languages_used = []

    num_angle = 0  
    # Generate text if needed
    if content_type in ["Text", "Both"]:
        size1 = random.randint(5, 12)
        unicode_ranges = [(33, 126), (0x0400, 0x04FF), (0x0370, 0x03FF), (0x4E00, 0x4E80)]
        languages = ["Latin", "Cyrillic", "Greek", "Chinese"]
        language_count = random.choices([1, 2], weights=[0.9, 0.1])[0]
        indices = np.arange(len(unicode_ranges))
        weights = [0.7, 0.1, 0.1, 0.1]
        chosen_indices = np.random.choice(indices, size=language_count, replace=False, p=weights)
        ranges = [unicode_ranges[i] for i in chosen_indices]
        if language_count == 1:
            for _ in range(size1):
                text += safe_char_from_range(ranges[0])
        else:
            for i in range(size1):
                text += safe_char_from_range(ranges[i < size1 // 2])
        languages_used = [languages[i] for i in chosen_indices]

    # Load logo if needed
    if content_type in ["Logo", "Both"]:
        logo_path = os.path.join(logo_dir, random.choice(logo_files))
        logo_img = crop_transparent_padding(logo_path)
        if(gray_scale):
            logo_img = convert_to_grayscale(logo_img)
    font_path = FONT_PATH
    numerical_size = {
        "Small": random.uniform(0.5, 0.6),
        "Medium": random.uniform(0.7, 0.8),
        "Large": random.uniform(0.9, 1.0)
    }[size]
    font_size = int(14 * numerical_size)
    ###########################################################
    ##opacity is set here
    opacity = {
        "Transparent": random.uniform(0.6, 0.7),
        "Semi-Transparent": random.uniform(0.7, 0.8),
        "Opaque": random.uniform(0.8, 1.0)
    }[appearance]
    ###########################################################
    num_angle1 = random.randint(-45, -30)if angle == "Inclined" else 0
    num_angle2 = random.randint(30, 45)if angle == "Inclined" else 0
    choose = random.randint(1, 2)
    num_angle = num_angle1 if choose == 1 else num_angle2

    # Determine region size for safe placement
    if content_type == "Both":
        logo_w, logo_h = logo_img.size
        pil_font = ImageFont.truetype(font_path, font_size)
        bbox = pil_font.getbbox(text)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if(logo_h > 3*text_h or logo_h < 0.8 * text_h):
            target_logo_height = int(text_h * random.uniform(2, 3))
            target_logo_width = int(logo_w * (target_logo_height / logo_h))
            logo_img = logo_img.resize((target_logo_width, target_logo_height), Image.Resampling.LANCZOS)
            # Match logo height to text height with a slight multiplier (optional)
            region_w = target_logo_width + text_w + 10
            region_h = max(target_logo_height, text_h)

    elif content_type == "Text":
        pil_font = ImageFont.truetype(font_path, font_size)
        bbox = pil_font.getbbox(text)
        region_w, region_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    elif content_type == "Logo":
        logo_w, logo_h = logo_img.size
        logo_scale = {
            "Small": random.uniform(0.1, 0.15),#logo sizing done here
            "Medium": random.uniform(0.15, 0.2),
            "Large": random.uniform(0.2, 0.25)
        }[size]
        new_logo_width = int(w * logo_scale)
        new_logo_height = int(logo_h * (new_logo_width / logo_w))
        logo_img = logo_img.resize((new_logo_width, new_logo_height), Image.Resampling.LANCZOS)
        region_w, region_h = logo_img.size

    # Repetitive pattern
    if location == "Repetitive":
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        
        if content_type == "Text":
            pil_font = ImageFont.truetype(font_path, font_size)
            bbox = pil_font.getbbox(text)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            #x_step = int(w * random.uniform(0.1, 0.5))
            #y_step = int(text_h * random.uniform(0.1, 0.2))
            #gap = int(text_w * 0.5)
            text_hypotenuse = int(sqrt(text_h**2 + text_w**2))
            x_step = text_hypotenuse // 2
            y_step = text_hypotenuse // 2
            gap = text_hypotenuse // 4

            if (pattern[0] == "Diamond" and angle == "Inclined"):
                if num_angle < 0:
                    start_x = 2 * int(-(h / tan(radians(-num_angle))))
                    if(h > w):
                        start_x *= 2
                    for i, x in enumerate(range(start_x, w, x_step)):
                        curr_x = x
                        for j, y in enumerate(range(0, h, int(y_step + gap))):
                            image = put_unicode_text(image, text, (curr_x, y), font_path, font_size, color, num_angle, opacity)
                            curr_x += x_step + gap
                elif num_angle > 0:
                    start_x = w + 2 * int(h / tan(radians(num_angle)))
                    if(h > w):
                        start_x *= 2
                    for i, x in enumerate(range(start_x, 0, -x_step)):
                        curr_x = x
                        for j, y in enumerate(range(0, h, int(y_step + gap))):
                            image = put_unicode_text(image, text, (curr_x, y), font_path, font_size, color, num_angle, opacity)
                            curr_x -= x_step + gap
            else:
                for i, y in enumerate(range(0, h, y_step)):
                    x_start = 0 if pattern == "Grid" or i % 2 == 0 else int(x_step / 2)
                    for x in range(x_start, w, x_step):
                        image = put_unicode_text(image, text, (x, y), font_path, font_size, color, num_angle, opacity)


        elif content_type == "Logo":
            # Convert OpenCV image to PIL RGBA
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
            # Prepare logo with new alpha (no blending)
            logo_img_clean = logo_img.copy()
            r, g, b, a = logo_img_clean.split()
            new_alpha = a.point(lambda p: int(p * opacity))
            logo_img_clean.putalpha(new_alpha)# Multiply existing alpha with desired opacity
            min_x_step,min_y_step = logo_img_clean.size
            if(angle == "Inclined"):
                x = 0
                y = 0
                logo_w,logo_h = logo_img_clean.size
                ##in this part minumum y difference is calulated by geometry.
                #first x and y differences of corners of the logo is found
                #then numerical value is calculated from geometic formula.
                center = np.array([x + logo_w/2,y + logo_h/2])
                topleft = np.array([x,y])
                topright = np.array([x+logo_w,y])
                bottom_left = np.array([x,y+logo_h])
                rotated_topleft = rotate_point_funct(topleft,center,num_angle)
                rotated_bottom_left = rotate_point_funct(bottom_left,center,num_angle)
                rotated_topright = rotate_point_funct(topright,center,num_angle)
                x_diff = abs(rotated_topleft[0]-rotated_bottom_left[0])
                y_diff = abs(rotated_topleft[1]-rotated_bottom_left[1]) 
                min_y_step = pow(x_diff,2)/y_diff
                min_y_step += y_diff
                x_diff = abs(rotated_topright[0]-rotated_topleft[0])
                y_diff = abs(rotated_topright[1]-rotated_topleft[1])
                min_x_step = sqrt(pow(x_diff,2)+ pow(y_diff,2))
                logo_img_clean = logo_img_clean.rotate(num_angle,resample=Image.BICUBIC,expand=True)

            x_step = int(min_x_step * random.uniform(1, 1.5))
            y_step = int(min_y_step * random.uniform(1, 1.5))
            gap = int(min_x_step*random.uniform(0.1,0.25))
            if (pattern[0] == "Diamond" and angle == "Inclined"):
                if num_angle < 0:
                    start_x = 2 * int(-(h / tan(radians(-num_angle))))
                    for i, x in enumerate(range(start_x, w, x_step)):
                        curr_x = x
                        for j, y in enumerate(range(0, h, int(min_y_step + gap))):
                            image_pil.paste(logo_img_clean, (int(curr_x), int(y)), logo_img_clean)
                            curr_x += x_step + gap
                elif num_angle > 0:
                    start_x = w + 2 * int(h / tan(radians(num_angle)))
                    for i, x in enumerate(range(start_x, 0, -x_step)):
                        curr_x = x
                        for j, y in enumerate(range(0, h, int(min_y_step + gap))):
                            image_pil.paste(logo_img_clean, (int(curr_x), int(y)), logo_img_clean)
                            curr_x -= x_step + gap
            else:
                for i, y in enumerate(range(0, h, y_step)):
                    x_start = 0 if pattern == "Grid" or i % 2 == 0 else int(x_step / 2)
                    for x in range(x_start, w, x_step):
                        image_pil.paste(logo_img_clean, (x, y), logo_img_clean)
                


            # FINAL: Flatten onto white background (to prevent black halo when saving)
            white_bg = Image.new("RGBA", image_pil.size, (255, 255, 255, 255))
            flattened = Image.alpha_composite(white_bg, image_pil)

            # Convert back to OpenCV
            image = cv2.cvtColor(np.array(flattened.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif content_type == "Both":
            # Clone logo and adjust its alpha by scaling original
            logo_img_clean = logo_img.copy()
            r, g, b, a = logo_img_clean.split()
            new_alpha = a.point(lambda p: int(p * opacity))
            logo_img_clean.putalpha(new_alpha)

            # Prepare font and text image
            pil_font = ImageFont.truetype(font_path, font_size)
            ascent, descent = pil_font.getmetrics()
            bbox = pil_font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = ascent + descent

            text_img = Image.new("RGBA", (text_w, text_h), (255, 255, 255, 0))
            draw = ImageDraw.Draw(text_img)
            draw.text(
                (0, 0),
                text,
                font=pil_font,
                fill=color + (int(opacity * 255),),
            )
            angle_rad = np.radians(num_angle)
            combined_w = (logo_img_clean.width + text_w)*np.cos(angle_rad)
            combined_h = (logo_img_clean.height + text_h)*np.cos(angle_rad)
            x_step = int(combined_w * random.uniform(1.5, 1.8))
            y_step = int(combined_h * random.uniform(1.5, 1.8))
            for i, y in enumerate(range(0, h, y_step)):
                x_start = 0 if pattern == "Grid" or i % 2 == 0 else int(x_step / 2)
                for x in range(x_start, w, x_step):##the same logic is applied here
                    x = int(x)
                    y = int(y)
                    center = (x + logo_img_clean.width // 2,y + logo_img_clean.height // 2)
                    text_pos = np.array([center[0] + logo_img_clean.width // 2, center[1] - text_img.height // 2])
                    logo_pos = np.array([x, y])

                    # --- Shared center of virtual combined image ---
                    combined_width = logo_img_clean.width + text_img.width
                    combined_height = max(logo_img_clean.height, text_img.height)
                    shared_center = np.array([x + combined_width // 2, y + combined_height // 2])

                    def rotate_point(pt, center, angle_deg):
                        angle_rad = np.radians(-angle_deg)
                        rotation_matrix = np.array([
                            [np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad),  np.cos(angle_rad)]
                        ])
                        return (rotation_matrix @ (pt - center)) + center

                    if angle == "Inclined":
                        # Rotate logo
                        logo_center_abs = logo_pos + np.array(logo_img_clean.size) / 2
                        rotated_logo_center = rotate_point(logo_center_abs, shared_center, num_angle)
                        rotated_logo = logo_img_clean.rotate(num_angle, resample=Image.BICUBIC, expand=True)
                        rotated_logo_size = np.array(rotated_logo.size)
                        rotated_logo_topleft = rotated_logo_center - rotated_logo_size / 2

                        # Rotate text
                        text_center_abs = text_pos + np.array(text_img.size) / 2
                        rotated_text_center = rotate_point(text_center_abs, shared_center, num_angle)
                        rotated_text = text_img.rotate(num_angle, resample=Image.BICUBIC, expand=True)
                        rotated_text_size = np.array(rotated_text.size)
                        rotated_text_topleft = rotated_text_center - rotated_text_size / 2
                        # Paste
                        image_pil.paste(rotated_logo, tuple(np.round(rotated_logo_topleft).astype(int)), rotated_logo)
                        image_pil.paste(rotated_text, tuple(np.round(rotated_text_topleft).astype(int)), rotated_text)
                    else:
                        # No rotation
                        image_pil.paste(logo_img_clean, tuple(logo_pos), logo_img_clean)
                        image_pil.paste(text_img, tuple(text_pos), text_img)

            # Composite everything over original image

            white_bg = Image.new("RGBA", image_pil.size, (255, 255, 255, 255))
            flattened = Image.alpha_composite(white_bg, image_pil)

            # Convert back to OpenCV
            image = cv2.cvtColor(np.array(flattened.convert("RGB")), cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_path, image)
        return languages_used,opacity


    # Safe placement area calculation
    safe_range = get_safe_rotated_position(region_w, region_h, num_angle, w, h)
    edge_distance = 0.1

    if location == "Corner":
        val1 = random.randint(0, 1)
        val2 = random.randint(0, 1)
        x = random.uniform(safe_range[0][0], safe_range[0][0] + safe_range[0][0]*0.05) if val1 == 0 else random.uniform(safe_range[0][1]*0.9, safe_range[0][1])
        y = random.uniform(safe_range[1][0], safe_range[1][0] + safe_range[1][0]*0.05) if val2 == 0 else random.uniform(safe_range[1][1]*0.9, safe_range[1][1])
    elif location == "Medium":
        x = random.uniform(safe_range[0][0]*edge_distance, safe_range[0][1]*(1-edge_distance))
        y = random.uniform(safe_range[1][0]*edge_distance, safe_range[1][1]*(1-edge_distance))
    
    # Rendering based on content
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    if content_type == "Text":
        image = put_unicode_text(image, text, (int(x), int(y)), font_path, font_size, color, num_angle, opacity)

    elif content_type == "Logo":
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        logo_img_clean = logo_img.copy()
        r, g, b, a = logo_img_clean.split()
        new_alpha = a.point(lambda p: int(p * opacity))
        logo_img_clean.putalpha(new_alpha)
        image_pil.paste(logo_img_clean,(int(x),int(y)), logo_img_clean)
        white_bg = Image.new("RGBA", image_pil.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(white_bg, image_pil)
        image = cv2.cvtColor(np.array(flattened.convert("RGB")), cv2.COLOR_RGB2BGR)

    elif content_type == "Both":
        # Base RGBA image from OpenCV
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # --- Prepare logo ---
        logo_img_clean = remove_premultiplied_alpha(logo_img.copy())
        r, g, b, a = logo_img_clean.split()
        new_alpha = a.point(lambda p: int(p * opacity))
        logo_img_clean.putalpha(new_alpha)

        # --- Prepare text ---
        pil_font = ImageFont.truetype(font_path, font_size)
        ascent, descent = pil_font.getmetrics()
        bbox = pil_font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = ascent + descent

        text_img = Image.new("RGBA", (text_w, text_h), (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_img)
        draw.text(
            (0, 0),
            text,
            font=pil_font,
            fill=color + (int(opacity * 255),),
        )

        # --- Determine initial positions ---
        x = int(x)
        y = int(y)
        center = (x + logo_img_clean.width // 2,y + logo_img_clean.height // 2)
        text_pos = np.array([center[0] + logo_img.width // 2, center[1] - text_img.height // 2])
        logo_pos = np.array([x, y])

        # --- Shared center of virtual combined image ---
        combined_width = logo_img_clean.width + text_img.width
        combined_height = max(logo_img_clean.height, text_img.height)
        shared_center = np.array([x + combined_width // 2, y + combined_height // 2])

        if angle == "Inclined":
            # Rotate logo
            logo_center_abs = logo_pos + np.array(logo_img_clean.size) / 2
            rotated_logo_center = rotate_point(logo_center_abs, shared_center, num_angle)
            rotated_logo = logo_img_clean.rotate(num_angle, resample=Image.BICUBIC, expand=True)
            rotated_logo_size = np.array(rotated_logo.size)
            rotated_logo_topleft = rotated_logo_center - rotated_logo_size / 2

            # Rotate text
            text_center_abs = text_pos + np.array(text_img.size) / 2
            rotated_text_center = rotate_point(text_center_abs, shared_center, num_angle)
            rotated_text = text_img.rotate(num_angle, resample=Image.BICUBIC, expand=True)
            rotated_text_size = np.array(rotated_text.size)
            rotated_text_topleft = rotated_text_center - rotated_text_size / 2
            # Paste
            image_pil.paste(rotated_logo, tuple(rotated_logo_topleft.astype(int)), rotated_logo)
            image_pil.paste(rotated_text, tuple(rotated_text_topleft.astype(int)), rotated_text)
        else:
            # No rotation
            image_pil.paste(logo_img_clean, tuple(logo_pos), logo_img_clean)
            image_pil.paste(text_img, tuple(text_pos), text_img)

        # --- Final flatten and convert ---
        white_bg = Image.new("RGBA", image_pil.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(white_bg, image_pil)
        image = cv2.cvtColor(np.array(flattened.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)
    return languages_used,opacity







metadata = []
image_id = 0
input_dir = INPUT_DIR
output_dir = OUTPUT_DIR
logo_dir = LOGO_DIR
logo_files = get_pngs_with_transparent_background(logo_dir)
for i in range(1,46):
    filename = os.path.join(input_dir, f"{i}.jpg")
    if os.path.exists(filename):
        out_path = os.path.join(output_dir,f"{i}.jpg" )
        content_type = random.choices(["Text", "Logo", "Both"],weights=[1,0,0])[0]
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX])
        location = random.choices(["Corner", "Medium", "Repetitive"], weights=[0,0,1])[0]
        pattern = random.choices(["Diamond", "Grid"],weights=[1,0]) if location == "Repetitive" else None
        appearance = random.choices(["Transparent", "Semi-Transparent", "Opaque"], weights=[0.4, 0.4,0.3])[0]
        size = random.choice(["Small", "Medium", "Large"])
        angle = random.choices(["Inclined","non-inclined"],weights=[1,0])[0]
        color = random.choices([(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)], weights=[0.7, 0.1, 0.1, 0.1])[0]
        gray_scale = random.choices([True,False],weights=[0.9,0.1])[0]
        language,opacity = applyWaterMark(filename,out_path,content_type=content_type,location=location,pattern=pattern,appearance=appearance,size=size,angle=angle,color=color,font=font,logo_files=logo_files,gray_scale=gray_scale)
        result,diff_ratio = compare_images(filename,out_path,1,0.05)
        metadata.append({
        "image_id": i,
        "content": {
            "type": content_type,
            "language": language,
            "font": font if content_type == "Text" else None
        },
        "location": location,
        "pattern": pattern if location == "Repetitive" else None,
        "appearance": (appearance,opacity),
        "size": size,
        "angle": angle,
        "color": color,
        "Success": result,
        "Difference Percentage": diff_ratio * 100,
        "GrayScale": gray_scale if content_type == "Logo" or content_type == "Both" else None
    })
        this_dir = THIS_DIR
        with open(os.path.join(this_dir, "metadata.json"), "w", encoding="utf-8") as f:json.dump(metadata, f, indent=4, ensure_ascii=False)