'''
<색상 분리>
cv2.split(m, mv=None) -> dst

• m : (BGR) 컬러 영상
• m v : 출력 영상
• dst : 출력 영상들의 리스트(m 이 3차원이면 3차원 list 로 반환)


<생상 결함>
cv2.split(m, mv=None) -> dst
• m v : 입력 영상 리스트(or 튜플)
• dst : 출력 영상
'''

import cv2
import numpy as np

src = cv2.imread('./images/lenna.jpg')
print('src shape', src.shape)

splits = cv2.split(src)

cv2.imshow('src', src)
cv2.imshow('B', splits[0])
cv2.imshow('G', splits[1])
cv2.imshow('R', splits[2])


########################################
## HSV 모델


cv2.waitKey()
cv2.destroyAllWindows()


