"""
@author: Knitwearbear
"""

import numpy as np
from grabscreen import grab_screen
import cv2
#import os
import time
from directkeys import PressKey, ReleaseKey, ReleaseAllKeys, W,A,S,D, SHIFT
from getkeys import key_check


def processImg(original):
    processed=cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    return processed

def detectHealthBar(image):
    lower=np.array([17,100,17])
    upper=np.array([50,200,50])
    mask=cv2.inRange(image, lower, upper)
    output=cv2.bitwise_and(image, image, mask=mask)
    return output
    
for i in list(range(5))[::-1]:
    print("Booting Raime... " + str(i+1))
    time.sleep(1)


paused=False
lastTime=time.time()
while(True):
    if not paused:
        screen=grab_screen(region=(0,28,799,627))
        filtered=detectHealthBar(screen)
        #screen=processImg(screen)
        #screen=cv2.resize(screen, (400, 300))
        cv2.imshow('window', filtered)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        PressKey(W)
        PressKey(SHIFT)
        #print(str(1/(time.time()-lastTime)) + " fps" )
        lastTime=time.time()
    keys=key_check()
    if 'T' in keys:
        if paused:
            paused = False
            for i in list(range(5))[::-1]:
                print("Unpausing... " + str(i+1))
                time.sleep(1)
            print('Unpaused.')
        else:
            ReleaseAllKeys()
            print ('Paused.')
            paused=True
            time.sleep(1)








