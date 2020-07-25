from PIL import Image
from pytesseract import *
import re
import cv2

img = Image.open('./images/ocr_test.jpg')

text = pytesseract.image_to_string(img, lang='kor+eng')

print(text)