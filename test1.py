import numpy as np
import cv2

# loading the image
# org_image = cv2.imread("images/16.png")
org_image = cv2.imread("images/5.png")
# org_image = cv2.imread("images/7.JPG")

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

# Convert to graycsale
img_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

ret, gra_thresh = cv2.threshold(img_gray,115,200, cv2.THRESH_BINARY) 
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(gra_thresh, (3,3), 0) 

# Morphological gradient
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
gradient = cv2.morphologyEx(img_blur, cv2.MORPH_GRADIENT, kernel)
cv2.imshow('Morphological gradient', gradient)
cv2.waitKey()

saliency = cv2.saliency.StaticSaliencyFineGrained_create()

(success, saliencyMap) = saliency.computeSaliency(img_blur)
saliencyMap = (saliencyMap*255).astype("uint8")

cv2.imshow('saliencyMap', saliencyMap)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=saliencyMap, threshold1=100, threshold2=220) # Canny Edge Detection
# edges = cv2.Canny(image=thresh, threshold1=100, threshold2=220) # Canny Edge Detection

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

ad_thresh = cv2.adaptiveThreshold(edges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)

cv2.imshow('ad_thresh', ad_thresh)
cv2.waitKey(0)

# contours, hierarchy = cv2.findContours(ad_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

# cv2.imshow('contour', image)
# cv2.waitKey(0)

# Fill rectangular contours
contours, hierarchy  = cv2.findContours(ad_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cnt = cnt[0] if len(cnt) == 2 else cnt[1]
for contour in contours:
    # len_contour = cv2.arcLength(contour, True)
    len_contour = cv2.arcLength(contour, False)
    if (3000 > len_contour) and (len_contour > 50): 
        # cv2.drawContours(image_copy, [c], -1, (255,255,255), -1)
        cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('contour', image_copy)
cv2.waitKey(0)

cv2.destroyAllWindows()