import cv2
import numpy as np

from .color_filter import SkinColorFilter
from .dominant_color import get_dominant_color_image

class HandRemover:
    
    def __init__(self):
        self.dominant_color = None
    
    def process(self, image, is_cropped=False):
        return image