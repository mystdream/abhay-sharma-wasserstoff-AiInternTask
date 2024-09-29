# models/text_extraction_model.py
import cv2
from PIL import Image
import pytesseract

class TextExtractor:
    def __init__(self):
        self.text_extractor = pytesseract.pytesseract

    def extract_text(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        text = self.text_extractor.image_to_string(pil_image)
        text = ''.join(char for char in text if char.isprintable())
        return text.strip()
