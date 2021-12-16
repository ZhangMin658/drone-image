import numpy as np
import cv2
import imutils

# loading the image
# org_image = cv2.imread("images/16.png")
org_image = cv2.imread("images/5.png")
# org_image = cv2.imread("images/7.JPG")
# org_image = cv2.imread("images/8.png")
# org_image = cv2.imread("images/9.png")

dim = (800, 900)

# resize image
resized = cv2.resize(org_image, dim, interpolation = cv2.INTER_AREA)

# image copy
img_copy = resized.copy()

height, width, channels = img_copy.shape
x1 = int(width * 0.2)
x2 = int(width * 0.8)
y1 = 0
y2 = height

crop_img = img_copy[y1:y2, x1:x2] # Crop

# resize image
image = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
image_copy = image.copy()
# cv2.imshow('Original', image)
# cv2.waitKey(0)

# Convert to RGB
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

blur = cv2.GaussianBlur(image_rgb, (7, 7), 2)

# Convert to Gray
blur_gray = cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)

ret, thresh = cv2.threshold(blur_gray,135,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('blur', thresh)
cv2.waitKey()

edges = cv2.Canny(image=thresh, threshold1=100, threshold2=200, apertureSize = 3) # Canny Edge Detection

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

# # Fill rectangular contours
contours, hierarchy  = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

for contour in contours:
    # if cv2.contourArea(contour) > cv2.arcLength(contour, True):
        len_contour = 0.01*cv2.arcLength(contour, True)

        if (13 > len_contour) and (len_contour > 1): 
            approx = cv2.approxPolyDP(contour, len_contour, True)
            # detect the shapes.
            # Position for writing text
            x,y = approx[0][0]

            if len(approx) > 5:
                cv2.drawContours(image_copy, [approx], -1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image_copy, "Antena", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

cv2.imshow('contour', image_copy)
cv2.waitKey(0)

cv2.destroyAllWindows()