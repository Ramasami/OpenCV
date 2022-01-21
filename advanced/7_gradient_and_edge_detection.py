# %% Imports
import numpy as np
import cv2 as cv


# %% Value Settings
dir = "../Resources/"

# %% Read Image
image = cv.imread(dir + "Photos/cats.jpg")
cv.imshow("Image", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# %% Canny
canny = cv.Canny(gray, 125, 125)
cv.imshow("Canny", canny)

# %% Laplation
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian", lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

cv.imshow("SobelX", sobelx)
cv.imshow("SobelY", sobely)

combines_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow("Combined Sobel", combines_sobel)


# %%
cv.waitKey(0)
