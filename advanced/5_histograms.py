# %% Imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# %% Value Settings
dir = "../Resources/"

# %% Read Image
image = cv.imread(dir + "Photos/cats.jpg")
cv.imshow("Image", image)

# %%
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# %% Histogram
'''
Used to see the color intesities in an image
images = [all images]
channels = which channel to use, since gray only 0th index
mask = if any mask present, it will compute hstogram only on mask
histSize = no of bins to use to plot the histogram
range = range of all possible pixel values
'''

# %% Grayscale Histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(gray_hist)


# %% Masked Histogram
blank = np.zeros(image.shape[:2], dtype='uint8')
mask = cv.circle(
    blank.copy(), (image.shape[1]//2, image.shape[0]//2), 100, 255, -1)
masked_image = cv.bitwise_and(gray, gray, mask=mask)
masked_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
cv.imshow('Masked Grayscale', masked_image)
plt.figure()
plt.title('Masked Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.plot(masked_hist)

# %% Colored Histogram
color = ('b', 'g', 'r')
plt.figure()
plt.title('Colored Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
plt.xlim(0, 255)
for i, col in enumerate(color):
    hist = cv.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)

# %% Colored Histogram
color = ('b', 'g', 'r')
mask = cv.circle(
    blank.copy(), (image.shape[1]//2, image.shape[0]//2), 100, 255, -1)
plt.figure()
plt.title('Colored Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')
masked_image = cv.bitwise_and(image, image, mask=mask)
plt.xlim(0, 255)
cv.imshow("Masked Color", masked_image)
for i, col in enumerate(color):
    hist = cv.calcHist([image], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)


# %%
plt.show()
cv.waitKey(0)
