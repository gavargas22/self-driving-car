import numpy as np
from PIL import ImageGrab
import cv2
import time
import lane_detector
import lane_detection_functions
from getkeys import key_check
import os


def keys_to_output(keys):
    #[A, W, D] 
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    
    return output

file_name = 'training_data.npy'


if os.path.isfile(file_name):
    print("File exists, loading previos data")
    training_data = list(np.load(file_name))
else:
    print("File does not exist")
    training_data = []
    


def main():
    for i in list(range(4)) [::-1]:
        print(i+1)
        time.sleep(1)
    
    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
#            white_yellow = lane_detection_functions.select_white_yellow_hsv(screen)
        gray = lane_detection_functions.convert_gray_scale(screen)
        resized_image = cv2.resize(gray, (80, 60))
        
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([resized_image,output])
        
        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
        
#            edges = lane_detection_functions.detect_edges(gray)
#            smooth_edges = lane_detection_functions.apply_smoothing(edges, 5)
#            roi = lane_detection_functions.select_region(smooth_edges)
#            lines = lane_detection_functions.hough_lines(roi)
#            image_with_lines = lane_detection_functions.draw_lines(roi, lines)
#            left_line, right_line = lane_detection_functions.lane_lines(screen, lines)
        
#            image_with_lines = lane_detection_functions.draw_lane_lines(screen, lane_detection_functions.lane_lines(screen, lines))
        
#            cv2.imshow('window', image_with_lines)
            
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
main()