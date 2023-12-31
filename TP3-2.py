import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import random
import bisect
import imageio
import argparse
import numpy as np
import copy
import kornia
import timeit
import torch
from kornia.utils import create_meshgrid

class Box:
    def __init__(self, p1, p2,p3,p4,scale, angle):
        self.p1 = int(p1[0]), int(p1[1])
        self.p2 = int(p2[0]), int(p2[1])
        self.p3 = int(p3[0]), int(p3[1])
        self.p4 = int(p4[0]), int(p4[1])
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
    
    def rotate_box(self):
        angle_rad = np.radians(self.angle)

        # Calculate the center of the bounding box
        center_x = (self.p1[0] + self.p2[0] + self.p3[0] + self.p4[0]) / 4
        center_y = (self.p1[1] + self.p2[1] + self.p3[1] + self.p4[1]) / 4

        # Translate the bounding box points so that the center is at the origin
        translated_points = np.array([
            [self.p1[0] - center_x, self.p1[1] - center_y],
            [self.p2[0] - center_x, self.p2[1] - center_y],
            [self.p3[0] - center_x, self.p3[1] - center_y],
            [self.p4[0] - center_x, self.p4[1] - center_y]
        ])

        # Define the rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad), np.cos(angle_rad)]])

        # Rotate the translated bounding box points
        rotated_points = np.dot(rotation_matrix, translated_points.T).T

        # Translate the points back to their original position
        rotated_points += np.array([center_x, center_y])

        # Extract rotated points and assign them to the box as integer values
        self.p1 = (int(rotated_points[0][0]), int(rotated_points[0][1]))
        self.p2 = (int(rotated_points[1][0]), int(rotated_points[1][1]))
        self.p3 = (int(rotated_points[2][0]), int(rotated_points[2][1]))
        self.p4 = (int(rotated_points[3][0]), int(rotated_points[3][1]))

    

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



    
    def getImagePortion(self, image):

        points = [self.p1, self.p2, self.p3, self.p4]

        min_x = min(points, key=lambda t: t[0])[0]
        max_x = max(points, key=lambda t: t[0])[0]
        min_y = min(points, key=lambda t: t[1])[1]
        max_y = max(points, key=lambda t: t[1])[1]

        box = ((min_x, min_y), (max_x, max_y))
        x1, y1 = box[0]
        x2, y2 = box[1]

        return image[int(y1):int(y2), int(x1):int(x2)]

    def get_vertices(self):
        return [list(self.p1), list(self.p2), list(self.p3), list(self.p4)]    
    
    def crop_image(self, image, bbox):
        # Calculate the bounding box coordinates
        left, top = np.min(bbox, axis=0).astype(int)
        right, bottom = np.max(bbox, axis=0).astype(int)

        # Crop the image
        cropped_image = image[top:bottom, left:right, :]

        return cropped_image
    
    @staticmethod
    def rotate_image(img, particle,box_size=25):

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



    #### THE WHOLE FUCKING PROBLEM IS HERE
    def get_histograms(self, particles, frame, box_Size):
        
        images = []
     
        for particle in particles:
            rotation= self.box.rotate_image(frame, particle, box_size)
            images.append(rotation)
            #Make an image full of zeros of 100 by 100
    
        #portions = [self.box.getImagePortion(image) for image in images]

        histograms = [self.color_hist(p) for p in images]


        return histograms
    #define a factorial funcion

   
    
    def prediction_step(self, particles, variance):
        new_particles = []
     
        for p in particles:
            x = p[0]
            y = p[1]
        
            x += np.random.normal(0, variance)
            y += np.random.normal(0, variance)
            scale = p[2] + np.random.normal(0, 0.01)
            angle = p[3] + np.random.normal(0, 0.08)
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


    def run_particle_filter(self, box_size):
                
           
                frames = list(self.get_frames())
          

                particles = self.initialize_particles(500, 7)
                asd = []
                sdf = []
                best_particles = []
                boxy = self.box
        

          
                initial_box = copy.copy(self.box)
                first_hist = self.color_hist(initial_box.getImagePortion(frames[0]))
          
                for i in tqdm(range(1, len(frames)-1)):
                    
                    #Im gonna time every method to see which one is causing the delay
                    particles = self.prediction_step(particles, 5)
            


                    histograms = self.get_histograms(particles, frames[i], box_size)

        
                    distances = self.histogram_distance(first_hist, histograms)
                    
                  

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
                    boxy = boxa.applyRotation(-a_mean).applyScaling(s_mean)
                    sdf.append(boxy.get_vertices())
             


                return asd ,sdf, best_particles
    





center_x, center_y = 640/2, 480/2
box_size= 25
pf = ParticleFilter("escrime-4-3.avi", Box((center_x-box_size, center_y-box_size), (center_x+box_size, center_y-box_size), (center_x+box_size, center_y+box_size), (center_x-box_size, center_y+box_size),1,0))
pf_time_start = timeit.default_timer()
particles,boxes,bp = pf.run_particle_filter(box_size)
pf_time_end = timeit.default_timer()
frames = list(pf.get_frames())
frames_part = zip(frames, particles,boxes,bp)

print("particle filter time: ", pf_time_end - pf_time_start)

def main(save_gif=False):
    new_frames = []
    for frame, pt, vertices, bp in frames_part:

        #for p in pt:
        #    cv2.circle(frame, (p[0], p[1]), 2, (0, 0, 255), -1)
        cv2.polylines(frame, np.int32([vertices]), True, (0, 255, 0), 2)
         
        cv2.imshow('frame', frame)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        new_frames.append(frame_rgb.copy())
        
        if cv2.waitKey(24)==27:
            if cv2.waitKey(0)==27:
                break

    if save_gif:
        fps = 24
        imageio.mimsave('output.gif', new_frames, fps=fps, loop=0)


main(save_gif=True)


