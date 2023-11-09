

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = 'escrime-4-3.avi'

initial_area = (int(640/2), int(480/2))


def make_box(point, box_size=25):
    #Make the bounding box
    x1 = point[0] - box_size
    x2 = point[0] + box_size
    y1 = point[1] - box_size
    y2 = point[1] + box_size
    return (x1, y1), (x2, y2)


#Give me a function that crops an image given a certain box

def just_box(image, box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    return image[y_min:y_max, x_min:x_max]

c1, c2 = make_box(initial_area)


cap = cv2.VideoCapture(path)

init_image = cap.read()[1]
initial = just_box(init_image, make_box(initial_area))
final = None

while True:

    ret, frame = cap.read()
    if ret == False:
        break
    
    #Draw the bounding tracking box
    #cv2.rectangle(frame, c1,c2, (0,0,255), 2)
    
    #Draw the current frame
    #cv2.imshow('frame', frame)

    
    #27
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()



#Define a function c that returns a dict r g b with the histogran of N bins for each channel of an image


def color_hist(image, N=20):
    r = np.histogram(image[:,:,0], bins=N)[0]
    g = np.histogram(image[:,:,1], bins=N)[0]
    b = np.histogram(image[:,:,2], bins=N)[0]
    return {'r':r, 'g':g, 'b':b}


# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1),
# plt.bar(range(len(color_hist(initial)['r'])), color_hist(initial)['r'],color='red')
# plt.bar(range(len(color_hist(initial)['g'])), color_hist(initial)['g'],color='green')
# plt.bar(range(len(color_hist(initial)['b'])), color_hist(initial)['b'],color='blue')
# plt.xlabel('Bins')
# plt.ylabel('Number of pixels')
# plt.subplot(1, 2, 2),plt.imshow(initial)
# plt.show()





#Generate random particles

def initialize_particles(N, initial_position,variance):
    particles = []
    for i in range(N):
        x = np.random.normal(initial_position[0], variance)
        y = np.random.normal(initial_position[1], variance)
        particles.append((int(x), int(y), 1/N))
    return particles


particles = initialize_particles(100, initial_area, 10)

#Plot the particles on top of initial image

# plt.figure(figsize=(10,10))
# plt.imshow(init_image)
# plt.scatter([p[0] for p in particles], [p[1] for p in particles], color='green', s=2)
# plt.show()



def get_histograms(particles):
    boxes =  [make_box(p) for p in particles]
    histograms = [color_hist(just_box(init_image, b)) for b in boxes]

    return histograms


def histogram_distance(h1,h2):
    r1, g1, b1 = np.array(h1['r']), np.array(h1['g']), np.array(h1['b'])
    r2, g2, b2 = np.array(h2['r']), np.array(h2['g']), np.array(h2['b'])

    norm_term_r = 1/np.sqrt(np.mean(r1)*np.mean(r2)*len(r1)**2)
    norm_term_g = 1/np.sqrt(np.mean(g1)*np.mean(g2)*len(g1)**2)
    norm_term_b = 1/np.sqrt(np.mean(b1)*np.mean(b2)*len(b1)**2)


    r = np.sqrt(1-norm_term_r*np.sum(np.sqrt(r1*r2)))
    g = np.sqrt(1-norm_term_g*np.sum(np.sqrt(g1*g2)))
    b = np.sqrt(1-norm_term_b*np.sum(np.sqrt(b1*b2)))

    return r+g+b


initial_hist = color_hist(initial)

distances = []
for hist in get_histograms(particles):
    distances.append(histogram_distance(hist, initial_hist))

foo = [particles[i][2] * (1/distances[i]) for i in range(len(particles))]

foo = foo/sum(foo)

print(foo)