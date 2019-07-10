#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 00:40:24 2018

@author: abk
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import keras
import h5py
import cv2
import Tkinter as tk
import dlib
from imutils import face_utils
import math

from keras.models import Sequential, load_model

model = load_model('weightsbest.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

#Setting up camera
# setup capture
camera = cv2.VideoCapture(0)

# reduce frame size to speed it up
w = 640
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4) 

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle

def _create_circle_arc(self, x, y, r, **kwargs):
    if "start" in kwargs and "end" in kwargs:
        kwargs["extent"] = kwargs["end"] - kwargs["start"]
        del kwargs["end"]
    return self.create_arc(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle_arc = _create_circle_arc

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def gaze_track():
    global camera, model,b 
    
    canvas.delete(b)
    
    # get frame
    ret, frame = camera.read()
    
    # mirror the frame (my camera mirrors by default)
    frame = cv2.flip(frame, 1)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray_frame, 1)
    
    # loop over the face detections
    right_eye = [] 
    face = []
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray_frame, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        count = 0
            
        for (x, y) in shape:
            if (count in range(42,48)):
                right_eye.append([x,y])  
            if (count in [1,8,15, 29, 33, 51, 36, 41, 42, 47]):
                face.append([x,y])
            count+=1
    
    right_mid = int((right_eye[0][1]+right_eye[3][1])/2)
    right_dist = int(round_up_to_even((right_eye[3][0] - right_eye[0][0])/2))
    
    right = gray_frame[int(right_mid-right_dist/2):int(right_mid+right_dist/2),right_eye[0][0]:right_eye[0][0]+right_dist*2]
    
    image_input = cv2.resize(right,(30, 15), interpolation = cv2.INTER_LINEAR)
    
    train_data = image_input.astype('float32')/255.0
    train_data = train_data.reshape(-1, 15, 30, 1)
    
    gaze = model.predict(train_data)
    
    b = canvas.create_circle(gaze[0,0], gaze[0,1], 20, fill="blue", outline="#DDD", width=4)
    root.after(50, gaze_track)

root = tk.Tk()
root.attributes('-fullscreen', True)

canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=True)
b = canvas.create_circle(0,0, 20, fill="blue", outline="#DDD", width=4)

root.after(100, gaze_track)
root.mainloop()