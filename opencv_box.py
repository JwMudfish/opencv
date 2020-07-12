import numpy as np
import cv2


mouse_is_pressing = False
start_x, start_y = -1, -1


def mouse_callback(event, x, y, flags, param):

    global mouse_is_pressing, start_x, start_y

    img_result = img_color.copy()
    #img_result = img_color

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_is_pressing = True
        start_x, start_y = x, y
        cv2.circle(img_result, (x,y), 1, (0,255,0), -1)

        cv2.imshow('img_color', img_result)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_is_pressing:
            cv2.rectangle(img_result, (start_x, start_y), (x,y), (255,0,0), 3)
            cv2.imshow('img_color', img_result)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_is_pressing = False

        cv2.rectangle(img_result, (start_x, start_y), (x,y), (0,0,255), 3)
        
        img_cat = img_color[min(start_y,y) : max(start_y,y), min(start_x,x) : max(start_x,x)]

        
        #img_cat = cv2.cvtColor(img_cat, cv2.COLOR_BGR2GRAY)
        #img_cat = cv2.cvtColor(img_cat, cv2.COLOR_GRAY2BGR)

        img_result[min(start_y,y) : max(start_y,y), min(start_x,x) : max(start_x,x)] = img_cat
        cv2.imshow('img_color', img_result)
        cv2.imshow('img_cat', img_cat)


img_color = cv2.imread('./images/apple.jpg')
#img_color = np.zeros((500,500,3), np.uint8)

cv2.imshow('img_color', img_color)

cv2.setMouseCallback('img_color', mouse_callback)


cv2.waitKey()
cv2.destroyAllWindows()

'''

import numpy as np
import cv2
from random import shuffle
import math

drawing = False
xi, yi = -1, -1
B = [i for i in range(256)]
G = [i for i in range(256)]
R = [i for i in range(256)]

def onMouse(event, x, y, flags, frame):
    global xi, yi, drawing, B, G, R

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        xi, yi = x, y
        shuffle(B), shuffle(G), shuffle(R)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(frame, (xi, yi), (x, y), (B[0], G[0], R[0]), 3)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(frame, (xi, yi), (x, y), (B[0], G[0], R[0]), 3)



frame = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', onMouse, param=frame)

while True:
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
'''