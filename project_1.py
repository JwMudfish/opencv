# 특정 폴더 안에 있는 이미지를 자동 재생하는 slide show 만들기
# 특정 폴더 안에 이미지를 5장 넣자
# 이미지는 일정한 시간이 지나면 다음 이미지로 변경
# 마지막 이미지에서는 처음 이미지로 돌아가기
# 중간에 ESC 를 누르지 않으면 무한반복
# ESC 키를 누르면 slide show 끝

import cv2
import numpy as np
import glob
import time

img_list = glob.glob('./images/*.jpg')

ob = 0
while True:    
    img = cv2.imread(img_list[ob])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1000,1000)
    cv2.imshow('image',img)

    ob = ob + 1
    if ob == len(img_list): ob = 0 

    time.sleep(1)
    if cv2.waitKey(1000) == 27:
        break
        

cv2.destroyAllWindows()
