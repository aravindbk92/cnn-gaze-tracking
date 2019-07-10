#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:02:07 2018

@author: abk
"""

import dlib
from imutils import face_utils
import cv2
import numpy as np
import math
import Tkinter as tk

count_h = 0
count_v = 0

image_data = np.empty((0,15,30))
mouse_data = np.empty((0,2))
face_data = np.empty((0,10,2))

font = cv2.FONT_HERSHEY_SIMPLEX

#Setting up dlib detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./data/shape_predictor_68_face_landmarks.dat')

#Setting up camera
# setup capture
camera = cv2.VideoCapture(0)

# reduce frame size to speed it up
w = 640
camera.set(cv2.CAP_PROP_FRAME_WIDTH, w) 
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, w * 3/4)    
    
    
def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def save_data():
    global image_data, mouse_data, face_data
    np.save("image_data", image_data)
    np.save("mouse_data", mouse_data)
    np.save("face_data", face_data)
    
def get_gaze_data(rect_x, rect_y):
    global camera, image_data, mouse_data, face_data
    
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
        # 36-41 left eye (36, 39)
        # 42-47 right eye (42, 45)
        # 1 - left face
        # 15 - righ face
        # 8 - chin
        # 29 - nose middle
        # 33 - nose bottom
        # 51 - mouth top
            
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
    image_data = np.append(image_data,image_input.reshape(-1,15,30), axis= 0)
    
    mouse_position = np.array([rect_x,rect_y]).reshape(-1,2)
    mouse_data = np.append(mouse_data,mouse_position, axis=0)    
    
    face = np.array(face).reshape(-1,10,2)
    face_data = np.append(face_data,face, axis=0)
    
def move_rectangle():
    global count_h, count_v
    if count_v == 11:
        #save_data()
        # clean up
        cv2.destroyAllWindows()
        camera.release()
        cv2.waitKey(1)
        root.destroy()
    if count_v != 11:
        if count_h < 19:
            canvas.move(b, 100, 0)
        elif count_h == 19:
            canvas.move(b, 0, 100)
            count_v += 1
        elif count_h > 19 and count_h < 38:
            canvas.move(b, -100, 0)
        elif count_h == 38:
            canvas.move(b, 0, 100)
            count_v += 1
            count_h = 0
    
        count_h += 1
        
        get_gaze_data(canvas.coords(b)[0]+50,canvas.coords(b)[1]+50)    
            
        root.after(100, move_rectangle)

root = tk.Tk()
root.attributes('-fullscreen', True)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

print screen_width,screen_height

canvas = tk.Canvas(root)
canvas.pack(fill='both', expand=True)
b = canvas.create_rectangle(-100, 0, 0, 100, fill='blue')

move_rectangle()
root.mainloop()