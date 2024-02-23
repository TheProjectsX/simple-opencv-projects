"""
Download Tesseract-OCR for Windows PC from Here: https://github.com/UB-Mannheim/tesseract/wiki
"""
import cv2
import pytesseract
import os

# Add the Tesseract-OCR exe file path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# Get Text
def parseText(image):
    text = pytesseract.image_to_string(image)

    return text


# Format Image
def formatImage(image):
    img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # img = cv2.medianBlur(img, 5)

    return img


# Get the Image Source
imgSource = input("\nEnter Image Path:> ")
if not os.path.isfile(imgSource):
    exit("\nFile Not Found!")

image = cv2.imread(imgSource)
image = formatImage(image)

# cv2.imshow("Preview", image)
# cv2.waitKey(0)
text = parseText(image)

print("\nParsed Text:\n")
print(text)
