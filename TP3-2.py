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
import numpy as np
import copy


class Box:
    def __init__(self, p1, p2,p3,p4,scale, angle):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.scale = scale
        self.angle = angle


    def applyScaling(self, factor):
        # Calculate the center of the quadrilateral
        center_x = (self.p1[0] + self.p2[0] + self.p3[0] + self.p4[0]) // 4
        center_y = (self.p1[1] + self.p2[1] + self.p3[1] + self.p4[1]) // 4

        # Scale each point relative to the center
        self.p1 = (
            int(center_x + (self.p1[0] - center_x) * factor),
            int(center_y + (self.p1[1] - center_y) * factor)
        )
        self.p2 = (
            int(center_x + (self.p2[0] - center_x) * factor),
            int(center_y + (self.p2[1] - center_y) * factor)
        )
        self.p3 = (
            int(center_x + (self.p3[0] - center_x) * factor),
            int(center_y + (self.p3[1] - center_y) * factor)
        )
        self.p4 = (
            int(center_x + (self.p4[0] - center_x) * factor),
            int(center_y + (self.p4[1] - center_y) * factor)
        )

        return self


    def applyRotation(self, angle):
        # Calculate the center of the quadrilateral
        center_x = (self.p1[0] + self.p2[0] + self.p3[0] + self.p4[0]) // 4
        center_y = (self.p1[1] + self.p2[1] + self.p3[1] + self.p4[1]) // 4

        # Translate each point to the origin
        translated_p1 = (self.p1[0] - center_x, self.p1[1] - center_y)
        translated_p2 = (self.p2[0] - center_x, self.p2[1] - center_y)
        translated_p3 = (self.p3[0] - center_x, self.p3[1] - center_y)
        translated_p4 = (self.p4[0] - center_x, self.p4[1] - center_y)

        # Rotate each point around the origin
        rotated_p1 = (
            int(translated_p1[0] * np.cos(angle) - translated_p1[1] * np.sin(angle)),
            int(translated_p1[0] * np.sin(angle) + translated_p1[1] * np.cos(angle))
        )
        rotated_p2 = (
            int(translated_p2[0] * np.cos(angle) - translated_p2[1] * np.sin(angle)),
            int(translated_p2[0] * np.sin(angle) + translated_p2[1] * np.cos(angle))
        )
        rotated_p3 = (
            int(translated_p3[0] * np.cos(angle) - translated_p3[1] * np.sin(angle)),
            int(translated_p3[0] * np.sin(angle) + translated_p3[1] * np.cos(angle))
        )
        rotated_p4 = (
            int(translated_p4[0] * np.cos(angle) - translated_p4[1] * np.sin(angle)),
            int(translated_p4[0] * np.sin(angle) + translated_p4[1] * np.cos(angle))
        )

        # Translate each point back to its original position
        self.p1 = (rotated_p1[0] + center_x, rotated_p1[1] + center_y)
        self.p2 = (rotated_p2[0] + center_x, rotated_p2[1] + center_y)
        self.p3 = (rotated_p3[0] + center_x, rotated_p3[1] + center_y)
        self.p4 = (rotated_p4[0] + center_x, rotated_p4[1] + center_y)

        return self

    def getBoundingBox(self):
        points = [self.p1, self.p2,self.p3,self.p4]

        min_x = min(points, key=lambda t: t[0])[0]
        max_x = max(points, key=lambda t: t[0])[0]
        min_y = min(points, key=lambda t: t[1])[1]
        max_y = max(points, key=lambda t: t[1])[1]

        return (min_x, min_y), (max_x, max_y)
    
    def getImagePortion(self, image):

        #Make and empty image using black pixels with the size of bounding box

        box = self.getBoundingBox()
        x1, y1 = box[0]
        x2, y2 = box[1]

        return image[int(y1):int(y2), int(x1):int(x2)]

    def get_vertices(self):
        return [list(self.p1), list(self.p2), list(self.p3), list(self.p4)]    



class ParticleFilter:

    def __init__(self, path, box):
        self.path = path
        self.box = box

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
    
    def plot_color_hist(self,image):
        hist = self.color_hist(image)
        plt.figure(figsize=(10,4))
        plt.plot(hist['r'], label='r')
        plt.plot(hist['g'], label='g')
        plt.plot(hist['b'], label='b')
        plt.show()


 
    def initialize_particles(self,N, variance):
        particles = []
        for i in range(N):
            x = self.box.p1[0]
            y = self.box.p1[1]
            x = np.random.normal(x, variance)
            y = np.random.normal(y, variance)
            particles.append((int(x), int(y), self.box.scale, self.box.angle,  1/N))
        return particles


    @staticmethod
    def color_hist(image, Nb=20):
        return cv2.calcHist([image], [0, 1, 2], None, [Nb, Nb, Nb], [0, 256, 0, 256, 0, 256])


    def get_histograms(self, particles, frame):
        boxes = [self.box.applyScaling(p[2]).applyRotation(p[3]) for p in particles]
        portions = [b.getImagePortion(frame) for b in boxes]
        histograms = [self.color_hist(p) for p in portions]
        return histograms

    
    def prediction_step(self, particles, variance):
        new_particles = []
     
        for p in particles:
            x = p[0]
            y = p[1]
        
            x += np.random.normal(0, variance)
            y += np.random.normal(0, variance)
            scale = p[2] + np.random.normal(0, 0.001)

            angle = p[3] + np.random.normal(0, 0.001)
            new_particles.append((int(x), int(y),scale,angle, 1/len(particles)))

        return new_particles
    

    @staticmethod
    def histogram_distance(href, particle_hist):
        distances = []
        for hist in particle_hist:
            distances.append(cv2.compareHist(href, hist, cv2.HISTCMP_BHATTACHARYYA))
        return distances
   

    @staticmethod
    def systematic_resampling(particles):
        N = len(particles)
        weights = np.array([particle[4] for particle in particles])
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
        sum_weights = sum([p[4] for p in resampled_particles])
        resampled_particles = [(p[0], p[1], p[2], p[3], p[4]/sum_weights) for p in resampled_particles]
        return resampled_particles
    

    @staticmethod
    def update_particles(particles, distances):
        new_weights = [1/(d + 1e-10) for d in distances]

        updated_particles = [(p[0], p[1], p[2],p[3],p[4] * new_weights[i]) for i, p in enumerate(particles)]
        #I want the sum of the updated particles to be 1
        sum_weights = sum([p[4] for p in updated_particles])
        updated_particles = [(p[0], p[1],p[2],p[3],p[4]/sum_weights) for p in updated_particles]
        return updated_particles

    def run_particle_filter(self):
                frames = list(self.get_frames())
                particles = self.initialize_particles(2, 5)
                asd = []
                sdf = []
                best_particles = []
                boxy = self.box
                box_size = 25
               
                initial_box = copy.copy(self.box)
                first_hist = self.color_hist(initial_box.getImagePortion(frames[0]))

              

                for i in tqdm(range(1, len(frames)-1)):

                    particles = self.prediction_step(particles, 5)
                 
                    distances = self.histogram_distance(first_hist, self.get_histograms(particles, frames[i]))
           
                    particles = self.update_particles(particles, distances)

                    particles = self.systematic_resampling(particles)
                

            
                    x_mean = np.sum([p[0]*p[4] for p in particles])
                    y_mean = np.sum([p[1]*p[4] for p in particles])
                    s_mean = np.sum([p[2]*p[4] for p in particles])
                    a_mean = np.sum([p[3]*p[4] for p in particles])
                  
                


                    best_particle = (int(x_mean), int(y_mean), s_mean, a_mean)

                    
                    asd.append(particles)
                  
                    best_particles.append(best_particle)

                    boxa = Box((best_particle[0]-box_size, best_particle[1]-box_size), (best_particle[0]+box_size, best_particle[1]-box_size), (best_particle[0]+box_size, best_particle[1]+box_size), (best_particle[0]-box_size, best_particle[1]+box_size),best_particle[2],best_particle[3])
                    boxy = boxa.applyScaling(s_mean).applyRotation(a_mean)
                    sdf.append(boxy.get_vertices())
                    
                return asd ,sdf, best_particles


center_x, center_y = 640/2, 480/2
box_size= 25
pf = ParticleFilter("escrime-4-3.avi", Box((center_x-box_size, center_y-box_size), (center_x+box_size, center_y-box_size), (center_x+box_size, center_y+box_size), (center_x-box_size, center_y+box_size),1,0))
particles,boxes,bp = pf.run_particle_filter()
frames = list(pf.get_frames())
frames_part = zip(frames, particles,boxes,bp)



def main(save_gif=False):
    new_frames = []
    for frame, pt, vertices, bp in frames_part:
       
      
        for p in pt:
            cv2.circle(frame, (p[0], p[1]), 2, (0, 0, 255), -1)
        cv2.polylines(frame, np.int32([vertices]), True, (0, 255, 0), 2)
         
        cv2.imshow('frame', frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        new_frames.append(frame_rgb.copy())
        
        if cv2.waitKey(40)==27:
            if cv2.waitKey(0)==27:
                break

    if save_gif:
        fps = 24
        imageio.mimsave('output.gif', new_frames, fps=fps, loop=0)



first_frame = frames[0]
boxo = Box((center_x-box_size, center_y-box_size), (center_x+box_size, center_y-box_size), (center_x+box_size, center_y+box_size), (center_x-box_size, center_y+box_size),1,0)
plt.figure(figsize=(10,4))
#Plot boxo 
plt.plot([boxo.p1[0], boxo.p2[0], boxo.p3[0], boxo.p4[0], boxo.p1[0]], [boxo.p1[1], boxo.p2[1], boxo.p3[1], boxo.p4[1], boxo.p1[1]], label='boxo')
boxo = boxo.applyRotation(20)
plt.imshow(boxo.getImagePortion(first_frame))
plt.show()

