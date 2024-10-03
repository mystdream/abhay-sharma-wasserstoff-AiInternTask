import cv2
from PIL import Image
import pytesseract

class TextExtractionModel:
    @staticmethod
    def extract_text(image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        text = pytesseract.image_to_string(pil_image)
        text = ''.join(char for char in text if char.isprintable())
        return text.strip()
