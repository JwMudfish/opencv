import cv2
import sys

img = cv2.imread('./images/apple.jpg')

# image load 에 실패한 경우 에러 메세지 출력
if img is None:
    print('Image load failed')
    sys.exit()

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 영상을 (200,200) 위치에 (400,400)의 크기로 영상 출력


cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 400,500)  # 처음 윈도우 생성시 cv2.WINDOW_NORMAL 로 생성해야만 동작
cv2.moveWindow('image', 200, 200)

cv2.imshow('image',img)

while True:
    if cv2.waitKey() == ord('q'):
        break


#cv2.imshow('gray_img', gray_img)

#cv2.waitKey()
cv2.destroyAllWindows()


#### subplot 구현

import matplotlib.pyplot as plt


