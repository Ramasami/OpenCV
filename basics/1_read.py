# In[1]:
import cv2 as cv

# In[2]:
dir = "../Resources/"

# In[3]:
# Reading Images

img = cv.imread(dir + 'Photos/cat.jpg')
cv.imshow('Cat', img)
cv.waitKey(0)

# In[4]:
# Reading Videos

capture = cv.VideoCapture(dir + 'Videos/dog.mp4')
while(capture.isOpened()):
    _, frame = capture.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
# In[5]:
# Reading Webcam

capture = cv.VideoCapture(0)
while(capture.isOpened()):
    _, frame = capture.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) == ord('d'):
        break
capture.release()
cv.destroyAllWindows()