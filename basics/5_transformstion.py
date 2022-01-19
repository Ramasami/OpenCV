# %% Imports
import numpy as np
import cv2 as cv


# %% Value Settings
dir = "../Resources/"

# %% Read Image
image = cv.imread(dir + "Photos/cats.jpg")
cv.imshow("Cats", image)
cv.waitKey(0)

# %% Translation
'''
-x -> left
x -> right
-y -> up
y -> down
'''


def translate(img, x, y):
    transMat = np.float32([[1, 1, x], [0, 1, y]])
    dimension = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimension)


translated = translate(image, -100, 100)
cv.imshow("Translated", translated)
cv.waitKey(0)

# %% Rotate
'''
angle -> ccw
-angle -> cw
'''


def rotate(img, angle, rotationPoint=None):
    h, w = img.shape[:2]
    if rotationPoint == None:
        rotationPoint = (w//2, h//2)
    rotMat = cv.getRotationMatrix2D(rotationPoint, angle, 1.0)
    print(rotMat)
    dimension = (w, h)
    return cv.warpAffine(img, rotMat, dimension)


rotated = rotate(image, 45)
cv.imshow("Rotated", rotated)
cv.waitKey(0)

cv.imshow("Rotated", rotate(rotate(image, 45), 45))
cv.waitKey(0)

rotated = image
step = 15
for i in range(0, 360, step):
    rotated = rotate(rotated, step)
cv.imshow("Rotated", rotated)
cv.waitKey(0)

# %% Resize
resized = cv.resize(image, (1000, 1000), interpolation=cv.INTER_CUBIC)
cv.imshow("Resized", resized)
cv.waitKey(0)

# %% Flipping
'''
0 -> flip verticaly
1 -> flip horizontally
'''
flip = cv.flip(image, 0)
cv.imshow("Flipped", flip)
cv.waitKey(0)

cv.imshow("Flipped", image[::-1,:,:])
cv.waitKey(0)


# %% Cropping
cv.imshow("Cropped", image[200:400,200:400])
cv.waitKey(0)
