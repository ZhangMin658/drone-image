import numpy as np
import cv2

# Read the image and perfrom an OTSU threshold
img = cv2.imread("images/16.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Remove hair with opening
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# Combine surrounding noise with ROI
kernel = np.ones((6,6),np.uint8)
dilate = cv2.dilate(opening,kernel,iterations=3)

# Blur the image for smoother ROI
blur = cv2.blur(dilate,(15,15))

# Perform another OTSU threshold and search for biggest contour
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('thresh', thresh)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea)

# Create a new mask for the result image
h, w = img.shape[:2]
mask = np.zeros((h, w), np.uint8)

# Draw the contour on the new mask and perform the bitwise operation
cv2.drawContours(mask, [cnt],-1, 255, -1)
res = cv2.bitwise_and(img, img, mask=mask)

# Display the result
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()