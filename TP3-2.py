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


class ParticleFilter:
    def __init__(self, path, initial_positions):
        self.path = path
        self.initial_positions = initial_positions

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
        # Stop the video if pressing the escape button
        if cv2.waitKey(30) == 27:
            if cv2.waitKey(0) == 27:
                return True
        return False

    @staticmethod
    def color_hist(image, Nb=20):
        return cv2.calcHist([image], [0, 1, 2], None, [Nb, Nb, Nb], [0, 256, 0, 256, 0, 256])

    def plot_color_hist(self, image):
        hist = self.color_hist(image)
        plt.figure(figsize=(10, 4))
        plt.plot(hist['r'], label='r')
        plt.plot(hist['g'], label='g')
        plt.plot(hist['b'], label='b')
        plt.show()

    @staticmethod
    def initialize_particles_multi(N, initial_positions, variance):
        particles = []
        for i in range(N):
            initial_states = [(np.random.normal(pos[0], variance), np.random.normal(pos[1], variance)) for pos in initial_positions]
            weights = [1 / N] * len(initial_positions)
            particles.append((initial_states, weights))
        return particles

    def prediction_step_multi(self, particles, variance):
        new_particles = []
        for p in particles:
            new_states = [(state[0] + np.random.normal(0, variance), state[1] + np.random.normal(0, variance)) for state in p[0]]
            new_particles.append((new_states, p[1]))
        return new_particles
    
    def make_multi_boxes(self, particles, box_size=25):
        return [self.make_box(p[0][i], box_size) for p in particles for i in range(len(p[0]))]


    def just_multi_boxes(self, image, boxes):
        return [self.just_box(image, box) for box in boxes]

    def get_multi_histograms(self, particles, frame):
        boxes = self.make_multi_boxes(particles)
        sub_images = self.just_multi_boxes(frame, boxes)
        histograms = [self.color_hist(sub_img) for sub_img in sub_images]
        return histograms


    def get_histograms_multi(self, particles, frame):
        boxes = self.make_multi_boxes(particles)
        sub_images = self.just_multi_boxes(frame, boxes)
        histograms = [self.color_hist(sub_img) for sub_img in sub_images]
        return histograms

    @staticmethod
    def histogram_distance_multi(particle_histograms, frame):
        distances = []
        for hist, box in zip(particle_histograms, self.make_multi_boxes(particle_histograms)):
            sub_img = self.just_box(frame, box)
            distances.append(cv2.compareHist(hist, self.color_hist(sub_img), cv2.HISTCMP_BHATTACHARYYA))
        return distances

    def update_particles_multi(self, particles, distances):
        new_weights = [1 / (d + 1e-10) for d in distances]
        updated_particles = [(p[0], [w * p[1][i] for i, w in enumerate(new_weights)]) for p in zip(particles, new_weights)]
        # Normalize weights
        sum_weights = [sum(p[1]) for p in updated_particles]
        updated_particles = [(p[0], [w / sum_weights[i] for w in p[1]]) for i, p in enumerate(updated_particles)]
        return updated_particles

    def systematic_resampling_multi(self, particles):
        N = len(particles)
        resampled_particles = []

        for i in range(N):
            u1 = np.random.uniform(0, 1 / N)
            u = u1 + np.arange(N) / N
            resampled_indices = np.searchsorted(np.cumsum(p[1]), u)
            resampled_states = [p[0][idx] for idx in resampled_indices]
            resampled_particles.append((resampled_states, [1 / N] * N))

        return resampled_particles

    def run_particle_filter_multi(self, box_size=25):
        frames = list(self.get_frames())
        particles = self.initialize_particles_multi(100, self.initial_positions, 5)
        asd = [(frames[0], self.initial_positions, particles)]

        for i in tqdm(range(1, len(frames) - 1)):
            particles = self.prediction_step_multi(particles, 5)
            distances = self.histogram_distance_multi(self.get_histograms_multi(particles, frames[i]), frames[i])
            particles = self.update_particles_multi(particles, distances)
            particles = self.systematic_resampling_multi(particles)

            asd.append((frames[i], [self.make_box(p[0], box_size=box_size) for p in particles], particles))

        return asd

def main_multi(path, save_gif=False, box_size=25):
    particleFilter = ParticleFilter(path, [(320, 240), (400, 100), (200, 300)])  # Add more initial positions as needed
    new_frames = []
    for frame, boxes, particles in particleFilter.run_particle_filter_multi(box_size=box_size)[0:300]:
        for box in boxes:
            cv2.rectangle(frame, box[0], box[1], (0, 0, 255), 2)

        for p in particles:
            for state in p[0]:
                cv2.circle(frame, (int(state[0]), int(state[1])), 1, (0, 255, 0), -1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        new_frames.append(frame_rgb.copy())
        cv2.imshow('frame', frame)
        if cv2.waitKey(40) == 27:
            if cv2.waitKey(0) == 27:
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
    main_multi(args.path, args.save_gif, args.box_size)
