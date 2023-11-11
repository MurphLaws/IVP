

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = 'escrime-4-3.avi'

initial_area = (int(640/2), int(480/2))



def make_box(point, box_size=20):
    #Make the bounding box
    x1 = point[0] - box_size
    x2 = point[0] + box_size
    y1 = point[1] - box_size
    y2 = point[1] + box_size
    return (x1, y1), (x2, y2)


def just_box(image, box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)


cap = cv2.VideoCapture(path)


#Read the first 10 frames of video

frames = []

for i in range(10):
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(frame)


first_frame = frames[0]
last_frame = frames[-1]


#Show the first and last frame side to side

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(first_frame)
plt.title('First Frame')

plt.subplot(1,2,2)
plt.imshow(last_frame)
plt.title('Last Frame')

plt.show()

#Save both images

cv2.imwrite('first_frame.png', first_frame)
cv2.imwrite('last_frame.png', last_frame)
