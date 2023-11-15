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
    def __init__(self, p1, p2,p3,p4,scale, angle):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.scale = scale
        self.angle = angle


    def applyScaling(self, factor):
        self.p1 = (int(self.p1[0]*factor), int(self.p1[1]*factor))
        self.p2 = (int(self.p2[0]*factor), int(self.p2[1]*factor))
        self.p3 = (int(self.p3[0]*factor), int(self.p3[1]*factor))
        self.p4 = (int(self.p4[0]*factor), int(self.p4[1]*factor))
     

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

        return image[y1:y2, x1:x2]

    def get_vertices(self):
        return [self.p1, self.p2, self.p3, self.p4]    



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


    
    def prediction_step(self, particles, variance):
        new_particles = []
     
        for p in particles:
            x = p[0]
            y = p[1]
        
            x += np.random.normal(0, variance)
            y += np.random.normal(0, variance)
            scale = p[2] + np.random.normal(0, 0.001)
            angle = p[3] + np.random.normal(0, variance)
            new_particles.append((int(x), int(y),scale, int(angle), 1/len(particles)))

        return new_particles
    

    #def get_histograms(self, particles, frame):
    #    return histograms

   
    def run_particle_filter(self):
                frames = list(self.get_frames())
                particles = self.initialize_particles(200, 5)
                asd = []
                sdf = []
                boxy = self.box
                for i in tqdm(range(1, len(frames)-1)):

                    particles = self.prediction_step(particles, 5)

                    #print(particles[0][2])
                    boxy = boxy.applyScaling(np.random.normal(1,0.01))
                    #print(boxy.get_vertices())
            
                    #distances = self.histogram_distance(first_hist, self.get_histograms(particles, frames[i]))
                    #particles = self.update_particles(particles, distances)


                #    particles = self.systematic_resampling(particles)
                
        
                 #   best_particle = particles[0]
                    asd.append(particles)
                    sdf.append(Box(boxy.p1,boxy.p2,boxy.p3,boxy.p4,boxy.scale,boxy.angle))
                    
                return asd  ,sdf


pf = ParticleFilter("escrime-4-3.avi", Box((200,195), (180, 184), (180, 200), (189, 174), 1, 0))
particles,boxes = pf.run_particle_filter()
frames = list(pf.get_frames())
frames_part = zip(frames, particles,boxes)


print([b.get_vertices() for b in boxes])

def main(path):
    for frame, pt,boxy in frames_part:
        

        vertices = np.array(boxy.get_vertices())
        
        vertices = vertices.reshape((-1,1,2))


        for p in pt:
            cv2.circle(frame, (p[0], p[1]), 1, (0, 255, 0), -1)
            cv2.polylines(frame, [vertices], True, (0, 255, 255), 2)



        cv2.imshow('frame', frame)
       
        if cv2.waitKey(40)==27:
            if cv2.waitKey(0)==27:
                break
main(2)

