# %% Imports

import cv2 as cv
import numpy as np

# %% Value Settings
dir = "../Resources/"

# %% Read File
img = cv.imread(dir + "/Photos/park.jpg")
cv.imshow("Cat", img)
cv.waitKey(0)

# %% GrayScale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)
cv.waitKey(0)

# %% Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow("Blur", blur)
cv.waitKey(0)

# %% Edge Cascade
canny = cv.Canny(img, 125,125)
cv.imshow("Canny", canny)
cv.waitKey(0)

canny = cv.Canny(blur, 125,125)
cv.imshow("Canny", canny)
cv.waitKey(0)

# %% Dilating the image
dilated = cv.dilate(canny, (3, 3), iterations=1)
cv.imshow("Dilated", dilated)
cv.waitKey(0)

dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow("Dilated", dilated)
cv.waitKey(0)

# %% Eroding
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow("Eroded", eroded)
cv.waitKey(0)

# %% Resize

resize = cv.resize(img, (500,500))
cv.imshow("Resized", resize)
cv.waitKey(0)

resize = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
cv.imshow("Resized", resize)
cv.waitKey(0)
# %% Crop
cropped = img[50:200,200:400]
cv.imshow("Cropped", cropped)
cv.waitKey(0)
