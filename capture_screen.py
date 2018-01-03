import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from directkeys import PressKey, W, A, S, D

def process_img(image):
    original_image = image
    # convert to gray
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edge detection
    processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
    return processed_img

def main():
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    last_time = time.time()
    while True:
        PressKey(W)
        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        #print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        new_screen = process_img(screen)
        cv2.imshow('window', new_screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
main()



#import numpy as np
#import pyscreenshot as ImageGrab
#import matplotlib.pyplot as plt
#from skimage import filters, color, feature
#from skimage.filters import roberts, sobel, scharr, prewitt
#
#if __name__ == "__main__":
#    while(True):
#        #    Grab the image of the screen
#        image = ImageGrab.grab(bbox=(1,20,800,600))
#        data = np.array(image, dtype=np.uint8)
#        
#        fig = plt.figure()
#        im = plt.imshow(data)
#        
#        im.set_data(data)
#        
#        plt.show()
        
    
#    ax.axis('off')
#    plt.show()
        
        
#    Convert to HSV
#    img_hsv = color.rgb2hsv(data)
    
    
#    img_hsv.show()
    
#while(True):
#    printscreen_pil =  ImageGrab.grab()
#    printscreen_numpy =   np.array(printscreen_pil.getdata(),dtype=uint8)\
#    .reshape((printscreen_pil.size[1],printscreen_pil.size[0],3)) 
#    cv2.imshow('window',printscreen_numpy)
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        break
