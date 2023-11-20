


import cv2
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from tqdm import tqdm
plot = lambda x: plt.imshow(x, cmap='gray').figure


img = cv2.imread("rotated_translated_smiley.png")
box_size = 50

def rotate(img, particle):

    x = particle[0]
    y = particle[1]
    rotation = particle[3]
    caja = np.array([[x-box_size, y-box_size], [x-box_size, y+box_size], [x+box_size, y+box_size], [x+box_size, y-box_size]])
    M = cv2.getRotationMatrix2D((x,y), rotation, 1)
    caja = np.array([np.dot(M, np.append(i, 1)) for i in caja])
    caja = np.array([[int(i[0]), int(i[1])] for i in caja])

    rect = cv2.minAreaRect(caja)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


rotated = rotate(img, [120, 80, 0, 45])


#Test the rotation function 1000 times and time it
import time

start = time.time()
for i in tqdm(range(1000)):
    rotated = rotate(img, [120, 80, 0, 45])

end = time.time()