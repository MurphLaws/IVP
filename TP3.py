import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import random
import bisect
import imageio
import argparse

class Box:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.p3 = (p1[0], p2[1])
        self.p4 = (p2[0], p1[1])



    def applyRotation(self, angle):
        angle = np.radians(angle)
        self.p1 = (self.p1[0]*np.cos(angle) - self.p1[1]*np.sin(angle), self.p1[0]*np.sin(angle) + self.p1[1]*np.cos(angle))
        self.p2 = (self.p2[0]*np.cos(angle) - self.p2[1]*np.sin(angle), self.p2[0]*np.sin(angle) + self.p2[1]*np.cos(angle))
        self.p3 = (self.p3[0]*np.cos(angle) - self.p3[1]*np.sin(angle), self.p3[0]*np.sin(angle) + self.p3[1]*np.cos(angle))
        self.p4 = (self.p4[0]*np.cos(angle) - self.p4[1]*np.sin(angle), self.p4[0]*np.sin(angle) + self.p4[1]*np.cos(angle))


    
    def applyScaling(self, factor):
        self.p1 = (self.p1[0]*factor, self.p1[1]*factor)
        self.p2 = (self.p2[0]*factor, self.p2[1]*factor)
        self.p3 = (self.p3[0]*factor, self.p3[1]*factor)
        self.p4 = (self.p4[0]*factor, self.p4[1]*factor)
    

class ParticleFilter:

    def __init__(self, path, initial_area):
        self.path = path
        self.initial_area = initial_area

    @staticmethod
    def make_box(point, box_size=25):
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
            if not ret or frame is None:
                break
            # Resize the frame to be 640 x 480
            frame = cv2.resize(frame, (640, 480)) if frame is not None else None
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
    def color_hist(image, Nb=20):
        return cv2.calcHist([image], [0, 1, 2], None, [Nb, Nb, Nb], [0, 256, 0, 256, 0, 256])


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

    def prediction_step(self, particles, variance):
        new_particles = []
     
        for p in particles:
            x = p[0]
            y = p[1]
            x += np.random.normal(0, variance)
            y += np.random.normal(0, variance)

            new_particles.append((int(x), int(y), p[2]))

        return new_particles

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
    def histogram_distance(href, particle_hist):
        distances = []
        for hist in particle_hist:
            distances.append(cv2.compareHist(href, hist, cv2.HISTCMP_BHATTACHARYYA))
        return distances

    @staticmethod
    def update_particles(particles, distances):
        new_weights = [1/(d + 1e-10) for d in distances]

        updated_particles = [(p[0], p[1], p[2] * new_weights[i]) for i, p in enumerate(particles)]
        #I want the sum of the updated particles to be 1
        sum_weights = sum([p[2] for p in updated_particles])
        updated_particles = [(p[0], p[1], p[2]/sum_weights) for p in updated_particles]
        return updated_particles

    @staticmethod
    def systematic_resampling(particles):
        N = len(particles)
        weights = np.array([particle[2] for particle in particles])
        cumsum_weights = np.cumsum(weights)
        u1 = np.random.uniform(0, 1/N)
        
        u = u1 + np.arange(N) / N
        resampled_particles = []
        j = 0

        for i in range(N):
            while u[i] > cumsum_weights[j]:
                j += 1
            resampled_particles.append(particles[j])

        #Return sorted_oarticles normalized
        sum_weights = sum([p[2] for p in resampled_particles])
        resampled_particles = [(p[0], p[1], p[2]/sum_weights) for p in resampled_particles]
        return resampled_particles

    @staticmethod
    def residual_resampling(particles):
        N = len(particles)  # Number of particles
        resampled_particles = []  # List to store resampled particles
        weights = [particle[2] for particle in particles]  # Extract weights from the particles
        cumulative_weights = [0.0] + list(np.cumsum(weights))

        for i in range(N):
            u = np.random.uniform(0, 1)
            sample_index = bisect.bisect_left(cumulative_weights, u * cumulative_weights[-1])
            resampled_particles.append(particles[sample_index - 1])

        for i in range(N):
            u = np.random.uniform(0, 1)
            fractional_weight = weights[i] - int(weights[i])  # Fractional part of the weight
            if u < fractional_weight:
                resampled_particles[i] = particles[i]

        sum_weights = sum([p[2] for p in resampled_particles])
        resampled_particles = [(p[0], p[1], p[2]/sum_weights) for p in resampled_particles]
        return resampled_particles


    @staticmethod
    def multinomial_resampling(particles):
        N = len(particles)  # Number of particles
        weights = np.array([particle[2] for particle in particles])  # Extract weights from the particles

        resampled_indices = np.random.choice(N, size=N, p=weights)
        resampled_particles = [particles[i] for i in resampled_indices]
        sum_weights = sum([p[2] for p in resampled_particles])
        resampled_particles = [(p[0], p[1], p[2]/sum_weights) for p in resampled_particles]
        return resampled_particles
    



    def run_particle_filter(self, box_size=25):
            frames = list(self.get_frames())
            particles = self.initialize_particles(500, self.initial_area, 5)
            asd = [(frames[0], self.initial_area, particles)]
            first_hist = self.color_hist(self.just_box(frames[0], self.make_box(self.initial_area, box_size=box_size)))
            
            for i in tqdm(range(1, len(frames)-1)):

                particles = self.prediction_step(particles, 5)
                distances = self.histogram_distance(first_hist, self.get_histograms(particles, frames[i]))
                particles = self.update_particles(particles, distances)


                particles = self.multinomial_resampling(particles)
            

                x_mean = np.sum([p[0]*p[2] for p in particles])
                y_mean = np.sum([p[1]*p[2] for p in particles])

                best_particle = (int(x_mean), int(y_mean))

               

                asd.append((frames[i], (best_particle[0], best_particle[1]),particles))

            return asd



# Create a new ParticleFilter instance





def main(path,save_gif=False, box_size=10):

    particleFilter = ParticleFilter(path, (320, 240))
    new_frames = []
    for frame, box, particles in particleFilter.run_particle_filter(box_size=box_size):
        box = particleFilter.make_box(box, box_size=box_size)
        cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)    
    
        for p in particles:
            cv2.circle(frame, (p[0], p[1]), 1, (0, 255, 0), -1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        new_frames.append(frame_rgb.copy())
        cv2.imshow('frame', frame)
        if cv2.waitKey(40)==27:
            if cv2.waitKey(0)==27:
                break

    if save_gif:
        fps = 24
        imageio.mimsave('output.gif', new_frames, fps=fps, loop=0)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run particle filter on a video and optionally save as GIF.")
    parser.add_argument("path", type=str, help="Path to the video file")
    parser.add_argument("--save_gif", action="store_true", help="Save the output as a GIF")
    parser.add_argument("--box_size", type=int, default=25, help="Size of the box around the object")

    args = parser.parse_args()
    main(args.path, args.save_gif, args.box_size)