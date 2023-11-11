import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk



class ParticleFilter:

    def __init__(self, path, initial_area):
        self.path = path
        self.initial_area = initial_area

    @staticmethod
    def make_box(point, box_size=20):
        x1 = point[0] - box_size
        x2 = point[0] + box_size
        y1 = point[1] - box_size
        y2 = point[1] + box_size
        return (x1, y1), (x2, y2)

    @staticmethod
    def just_box(image, box):
        x1, y1 = box[0]
        x2, y2 = box[1]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        return image[y_min:y_max, x_min:x_max]

    def get_frames(self):
        cap = cv2.VideoCapture(self.path)
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            yield frame


    @staticmethod
    def display(frame):
        cv2.imshow('frame', frame)
        #stop the video if pressing the escape button
        if cv2.waitKey(30)==27:
            if cv2.waitKey(0)==27:
                return True 
        return False


    
    @staticmethod
    def just_box(image, box):
        x1, y1 = box[0]
        x2, y2 = box[1]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        return image[y_min:y_max, x_min:x_max]


    @staticmethod
    def color_hist(image):
        return cv2.calcHist([image], [0, 1, 2], None, [20, 20, 20], [0, 256, 0, 256, 0, 256])


    def plot_color_hist(self,image):
        hist = self.color_hist(image)
        plt.figure(figsize=(10,4))
        plt.plot(hist['r'], label='r')
        plt.plot(hist['g'], label='g')
        plt.plot(hist['b'], label='b')
        plt.show()


    @staticmethod
    def initialize_particles(N, initial_position, variance):
        particles = []
        for i in range(N):
            x = np.random.normal(initial_position[0], variance)
            y = np.random.normal(initial_position[1], variance)
            particles.append((int(x), int(y), 1/N))
        return particles


    def get_histograms(self, particles, frame):
        boxes = [self.make_box(p) for p in particles]
        histograms = [self.color_hist(self.just_box(frame, b)) for b in boxes]
        return histograms



    @staticmethod
    def display_frame(frame):
        cv2.imshow('frame', frame)
        #stop the video if pressing the escape button
        if cv2.waitKey(0)==27:
            return True


    @staticmethod
    def systematic_resampling(particles):
        N = len(particles)
        cumulative_weights = np.cumsum([p[2] for p in particles])
        u = (np.arange(N) + np.random.uniform(0, 1)) / N
        indexes = np.searchsorted(cumulative_weights, u)

        resampled_particles = [(particles[i][0], particles[i][1], 1/N) for i in indexes]
        return resampled_particles

    @staticmethod
    def histogram_distance(href, particle_hist):
        distances = []
        for hist in particle_hist:
            distances.append(cv2.compareHist(href, hist, cv2.HISTCMP_BHATTACHARYYA))
        return distances

    @staticmethod
    def get_best_particle(particles, distances):
        new_weights = [1 / (d + 1e-10) for d in distances]

        updated_particles = [(p[0], p[1], p[2] * new_weights[i]) for i, p in enumerate(particles)]

        return sorted(updated_particles, key=lambda x: x[2], reverse=True)[0][:2]



    def run_particle_filter(self):
            frames = list(self.get_frames())
            asd = [(frames[0], self.initial_area)]
            
            for i in tqdm(range(1, len(frames)-1)):
                prior_box = self.just_box(asd[-1][0],self.make_box(asd[-1][1]))#just_box(frames[-1], make_box(asd[-1][1]))
                prior_hist = self.color_hist(prior_box)

                particles = self.initialize_particles(100, asd[-1][1], 5)

                distances = self.histogram_distance(prior_hist, self.get_histograms(particles, frames[i]))
                particles = self.systematic_resampling(particles)
                best_particle = self.get_best_particle(particles, distances)
                asd.append((frames[i], (best_particle[0], best_particle[1])))

            return asd



# Create a new ParticleFilter instance
particleFilter = ParticleFilter('escrime-4-3.avi', (320, 240))

for frame, box in particleFilter.run_particle_filter():
    box = particleFilter.make_box(box)
    cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)    
    cv2.imshow('frame', frame)
    if cv2.waitKey(30)==27:
        if cv2.waitKey(0)==27:
            break
