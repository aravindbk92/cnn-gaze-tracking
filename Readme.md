# CNN Eye Gaze Tracker
Quick and dirty implementation of eye gaze tracker using a CNN.

## Requirements:
Python 2.7
Keras
Numpy
cv2
h5py
Tkinter
tensorflow
imutils

## How it works

### Data collection (collect_data.py)
1. Shows a rectangle on the screen for the user to look at
2. Detects the face using a deformed face model
3. Crops the right eye and processes it
4. Associate eye crop with corresponding screen location of rectangle in dataset
5. Repeat from step 1 until needed

### Model training (train.py)
Trains using Keras. Architecture can be found in train.py.

### Eye gaze predicion (test.py)
Uses previously trained model to predict location on the screen where user is looking at (shown with a blue circle)