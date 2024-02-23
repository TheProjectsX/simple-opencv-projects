import cv2
import numpy as np
from PIL import Image, ImageDraw
import math

chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]


charArray = list(chars)
charLength = len(charArray)
interval = charLength/256

scaleFactor = 0.09

oneCharWidth = 10
oneCharHeight = 18


# Convert from Image to Ascii Art
# [Note]: Have no Idea about the Below function code nor the above variables. So don't time waste thinking what you wrote!

def getChar(inputInt):
    return charArray[math.floor(inputInt*interval)]


def image_to_ascii_art(image):
    img = Image.fromarray(image)
    width, height = img.size
    img = img.resize((int(scaleFactor*width), int(scaleFactor *
                     height*(oneCharWidth/oneCharHeight))), Image.NEAREST)
    width, height = img.size
    pix = img.load()

    outputImage = Image.new(
        'RGB', (oneCharWidth * width, oneCharHeight * height), color=(0, 0, 0))
    d = ImageDraw.Draw(outputImage)

    for i in range(height):
        for j in range(width):
            r, g, b = pix[j, i]
            h = int(r/3 + g/3 + b/3)
            pix[j, i] = (h, h, h)
            d.text((j*oneCharWidth, i*oneCharHeight),
                   getChar(h), fill=(r, g, b))

    outputImage = np.array(outputImage)
    outputImage = outputImage[:, :, ::-1].copy()

    return outputImage

# Convert the Color


def changeColor(image):

    # Define the lower and upper bounds of the color you want to change (in BGR format)
    lower_bound = np.array([20, 20, 20])
    upper_bound = np.array([255, 255, 255])

    # Create a mask for the specified color range
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Replace the pixels in the image with the new color (e.g., red in BGR format)
    image[mask > 0] = [0, 255, 0]

    return image


# Camera Input
cap_vid = cv2.VideoCapture(1)
cap_vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

while cap_vid.isOpened():
    ret, frame = cap_vid.read()
    if not ret:
        continue

    frame = cv2.flip(frame, cv2.CAP_PROP_XI_DECIMATION_HORIZONTAL)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ascii_image = image_to_ascii_art(frame)

    # color_changed = changeColor(ascii_image.copy())
    # cv2.imshow("Color Changed", color_changed)

    cv2.imshow("Ascii Mode", ascii_image)

    if (cv2.waitKey(1) & 0xFF == 27):
        break


cap_vid.release()
cv2.destroyAllWindows()
