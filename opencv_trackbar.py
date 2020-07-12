# Trackba r (트랙바) 란?
#    프로그램 실행 중 사용자가 변경하면서 영상을 변경하게 해주는 도구


# cv2.createTrackbar(trackbarName, windowName, value, count, onChange)
# • trackbarName : 트랙바 이름
# • windowName : 트랙바를 포함하는 윈도우 이름
# • value : 트랙바 초기값
# • count : 트랙바 최댓값(defalut = 0 )
# • onChange : 트랙바 값이 변경될떄마다 호출할 콜백 함수 이름

import cv2
import numpy as np

# grayscale 조절 함수
def level_change(pos):

    value = pos * 8

    if value >= 255:
        value = 255

    img[:] = value
    cv2.imshow('img', img)

#cv2.createTrackbar('lavel','img', 0, 32, level_change)

#cv2.imshow('image', img)

#cv2.waitKey()
#cv2.destroyAllWindows()


##########################################################################
# 트랙바를 이용히여 B,G,R 컬러를 각각 조절할 수 있는 함수

def fn(x):
    pass

# def rgb_change(pos):
#     b,g,r = cv2.split(img)
#     b = pos + 1
#     g = pos + 1
#     r = pos + 1

#     #b,g,r = [b ]

#     img[:] = [b,g,r]
#     cv2.imshow('image', img)



img = np.zeros((400, 400,3), dtype = np.uint8)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)


cv2.createTrackbar('R','image', 0, 255, fn)
cv2.createTrackbar('G','image', 0, 255, fn)
cv2.createTrackbar('B','image', 0, 255, fn)

while True: 
    cv2.imshow('image', img) 
    R = cv2.getTrackbarPos('R', 'image') 
    G = cv2.getTrackbarPos('G', 'image') 
    B = cv2.getTrackbarPos('B', 'image') 
    
    img[:] = [B,G,R] 
    
    k = cv2.waitKey(1) 
    
    if k == 27: 
        break 
    

cv2.destroyAllWindows()


