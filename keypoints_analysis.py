import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('Resources/mitchell.jpg')

# if image is None:
#     print("Image not found.")
#     exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)


for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area > 100:
        print(f"Area: {area:.2f}, Perimeter: {perimeter:.2f}")
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imwrite('output_with_contours.jpg', image_contours)
