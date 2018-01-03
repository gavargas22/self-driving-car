import numpy as np
from PIL import ImageGrab
import cv2
import time
import lane_detector
import lane_detection_functions
from getkeys import key_press
import os

def main():
    last_time = time.time()
    while True:
        try:
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 800, 640)))
            #print('Frame took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            
#            white_yellow = lane_detection_functions.select_white_yellow_hsv(screen)
            gray = lane_detection_functions.convert_gray_scale(screen)
            edges = lane_detection_functions.detect_edges(gray)
            smooth_edges = lane_detection_functions.apply_smoothing(edges, 5)
            roi = lane_detection_functions.select_region(smooth_edges)
            lines = lane_detection_functions.hough_lines(roi)
#            image_with_lines = lane_detection_functions.draw_lines(roi, lines)
#            left_line, right_line = lane_detection_functions.lane_lines(screen, lines)
            
            image_with_lines = lane_detection_functions.draw_lane_lines(screen, lane_detection_functions.lane_lines(screen, lines))
            
            cv2.imshow('window', image_with_lines)
            
        except:
            pass
            
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
main()