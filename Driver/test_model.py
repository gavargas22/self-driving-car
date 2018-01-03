import numpy as np
from PIL import ImageGrab
import cv2
import time
import lane_detector
import lane_detection_functions
from getkeys import key_check
import os
from alexnet import alexnet
from directkeys import PressKey, ReleaseKey, W, A, S, D

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)
    
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)

def right():
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    for i in list(range(4)) [::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    paused = False
    
    while True:
        
        if not paused:
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
            gray = lane_detection_functions.convert_gray_scale(screen)
            resized_image = cv2.resize(gray, (80, 60))
            
            prediction = model.predict([resized_image.reshape(WIDTH,HEIGHT,1)])[0]
            moves = list(np.around(prediction))
            print(moves, prediction)
            
            if moves == [1,0,0]:
                left()
            elif moves == [0,1,0]:
                straight()
            elif moves == [0,0,1]:
                right()
            
        keys = key_check()
        
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
        
main()