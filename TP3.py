

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


#Give me a function that crops an image given a certain box

def just_box(image, box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    return image[y_min:y_max, x_min:x_max]

c1, c2 = make_box(initial_area)


cap = cv2.VideoCapture(path)


frames = []

#Read the video frames and store them into frames list, dont show the video

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(frame)





def color_hist(image, N=20):
    r = np.histogram(image[:,:,0], bins=N)[0]
    g = np.histogram(image[:,:,1], bins=N)[0]
    b = np.histogram(image[:,:,2], bins=N)[0]
    return {'r':r, 'g':g, 'b':b}


def initialize_particles(N, initial_position,variance):
    particles = []
    for i in range(N):
        x = np.random.normal(initial_position[0], variance)
        y = np.random.normal(initial_position[1], variance)
        particles.append((int(x), int(y), 1/N))
    return particles



def get_histograms(particles,frame):
    boxes =  [make_box(p) for p in particles]
    histograms = [color_hist(just_box(frame, b)) for b in boxes]

    return histograms


#def histogram_distance(h1,h2):
#    return cv2.compareHist(h1['r'], h2['r'], cv2.HISTCMP_BHATTACHARYYA)
    
def histogram_distance(h1,h2):
    r1, g1, b1 = np.array(h1['r']), np.array(h1['g']), np.array(h1['b'])
    r2, g2, b2 = np.array(h2['r']), np.array(h2['g']), np.array(h2['b'])

    epsilon = 1e-10  # Small constant to avoid division by zero

    norm_term_r = 1/np.sqrt(np.mean(r1)*np.mean(r2)*len(r1)**2 + epsilon)
    norm_term_g = 1/np.sqrt(np.mean(g1)*np.mean(g2)*len(g1)**2 + epsilon)
    norm_term_b = 1/np.sqrt(np.mean(b1)*np.mean(b2)*len(b1)**2 + epsilon)

    r = np.sqrt(1-norm_term_r*np.sum(np.sqrt(r1*r2)))
    g = np.sqrt(1-norm_term_g*np.sum(np.sqrt(g1*g2)))
    b = np.sqrt(1-norm_term_b*np.sum(np.sqrt(b1*b2)))

    return r+g+b

def get_best_particle(particles):
    return particles[np.argmax(norm_weights)]





asd = [(frames[0], initial_area)]
boxes = []
for i in range(1,len(frames)):
    prior_box = just_box(frames[-1], make_box(asd[-1][1]))
    prior_hist = color_hist(prior_box)

    particles = initialize_particles(100, asd[-1][1], 2)
    distances = []

    for hist in get_histograms(particles, frames[i]):
        distances.append(histogram_distance(hist, prior_hist)+1e-9)

    foo = [particles[i][2] * (1/distances[i]) for i in range(len(particles))]
   
    norm_weights = foo/sum(foo)
    particles_coords = [p[:2] for p in particles]
    asd.append((frames[i], get_best_particle(particles_coords)))
  

 



while True:
    for frame, box in asd:
        box = make_box(box)
        cv2.rectangle(frame, box[0], box[1], (0,0,255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    break