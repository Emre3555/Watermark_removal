from PIL import ImageFont, ImageDraw, Image,ImageChops
from WaterMark import get_pngs_with_transparent_background
from test import crop_transparent_padding
import os
import numpy as np
def rotate_point(pt, center, angle_deg):
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    return (rotation_matrix @ (pt - center)) + center
pt = np.array([3,3])
center = np.array([2,2])
memet = rotate_point(pt,center,45)
print(memet)

