import cv2
import easyocr

class TextExtractionModel:
    def __init__(self)
        self.reader = easyocr.Reader(['en'])
    def extract_text(self, image_path):
        image = cv2.imread(image_path)
        results = self.reader.readtext(image)
        extracted_text = ' '.join([result[1] for result in results])
        return ''.join(char for char in extracted_text if char.isprintable()).strip()
