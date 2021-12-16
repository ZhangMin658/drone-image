import numpy as np
import cv2

# loading the image
# org_image = cv2.imread("img/16.png")
# org_image = cv2.imread("img/5.png")
org_image = cv2.imread("img/9.png")

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
cv2.imshow('Original', image)
cv2.waitKey(0)

# Convert to RGB
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# blur1 = cv2.GaussianBlur(image_rgb, (13, 13), 3)

# sigmaX is Gaussian Kernel standard deviation 
# ksize is kernel size 
blur = cv2.GaussianBlur(src=image_rgb, ksize=(13,13), sigmaX=0, sigmaY=0) # small noise

# # # medianBlur() is used to apply Median blur to image
# # # ksize is the kernel size
median = cv2.medianBlur(src=blur, ksize=3)

# sigmaSpace is used to filter sigma in the coordinate space.
sigma = cv2.bilateralFilter(src=median, d=7, sigmaColor=25, sigmaSpace=25) # small no closed 

cv2.imshow('filter', sigma)
cv2.waitKey(0)

# Convert to Gray
blur_gray = cv2.cvtColor(sigma,cv2.COLOR_RGB2GRAY)


# ret, thresh = cv2.threshold(blur_gray,195,215,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret, thresh = cv2.threshold(blur_gray,163,215,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('blur', thresh)
cv2.waitKey()

edges = cv2.Canny(image=thresh, threshold1=195, threshold2=215, apertureSize = 3) # Canny Edge Detection
# edges = cv2.Canny(image=thresh, threshold1=0, threshold2=100, apertureSize = 3) # Canny Edge Detection

# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

# # Fill rectangular contours
contours, hierarchy  = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image_copy, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)

for contour in contours:
    # if cv2.contourArea(contour) > cv2.arcLength(contour, True):
        len_contour = 0.01*cv2.arcLength(contour, True)

        # if (13 > len_contour) and (len_contour > 1): 
        if (len_contour > 1): 
            approx = cv2.approxPolyDP(contour, len_contour, True)
            # detect the shapes.
            # Position for writing text
            x,y = approx[0][0]

            if len(approx) > 5:
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Antena", (x, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

cv2.imshow('contour', image)
cv2.imwrite('result.jpg', image)
cv2.waitKey(0)

cv2.destroyAllWindows()