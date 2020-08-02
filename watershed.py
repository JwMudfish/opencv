

##### distanceTransform 함수 예제
'''
import cv2

img = cv2.imread('./images/ob/watershed_ex1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# image 가 너무 커서 resize 하기(resizing 을 하면 0.95 * dist_transform.max() 를 통해 threshold 할 때 작동 안 함)
#gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)

# 팽창 & 침식을 통해 노이즈 제거
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dst = cv2.dilate(gray, k)
erosion = cv2.erode(dst, k)

# 이진화된 결과를 dist_transform 함수의 입력으로 사용합니다.

dist_transform = cv2.distanceTransform(erosion, cv2.DIST_L2, 5)

# dist_transform 의 결과는 float32 이므로 imshow 를 위해서는 normalize 함수를 사용해야 합니다.
result = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

ret, thresh = cv2.threshold(result, 0.95 * dist_transform.max(), 255, cv2.THRESH_BINARY)

cv2.imshow("dist_transform", thresh)

cv2.waitKey(0)

'''


############ watershed 로 동전 윤곽선 구하기

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./images/ob/water_coins.jpg')

# grayscale 로 변환
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# opening 으로 noise 제거
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=2)

# dilate 를 통해서 확실한 Backgroud 찾기
sure_bg = cv2.dilate(opening, kernel, iterations = 3)

# distance transform 을 적용하면 중심으로 부터 Skeleton Image 를 얻자.
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# 이진화를 통해 확실한 foreground 파악하기 : max 에 곱하는 값함(= 0.5)은 계속 수정하며 확인해야
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Background 에서 Foregrand 를 제외한 영역을 Unknow 영역으로 파악
unknown = cv2.subtract(sure_bg, sure_fg)

# Foregrand 에 Labelling 작업 (cv2.connectedComponents 는 6강 강의자료 참고)
ret, markers = cv2.connectedComponents(sure_fg)
#print(markers)

# markers 에 1을 더해서 background 는 0 에서 1 로 변경됨
markers = markers + 1

# markers 에서 아직 정확하게 모르는 부분(unknown == 255 인 부분)을 0 으로 변경하자
markers[unknown == 255] = 0

# watershed 알고리즘 결과 : 동전 contour => 라벨 = -1 & 배경 => 라벨 = 1
markers = cv2.watershed(img, markers)
print(markers.shape)
print(markers)

# markers == -1 이면 contour 이므로, contour 를 빨간색으로 칠하자(plt 로 출력할 예정)
img[markers == -1] = [255, 0, 0]

images = [gray, thresh, sure_bg,  dist_transform, sure_fg, unknown, markers, img]
titles = ['Gray', 'Binary', 'Sure BG', 'Distance', 'Sure FG', 'Unknown', 'Markers', 'Result']

for i in range(len(images)):
    plt.subplot(2, 4, i+1),\
    plt.imshow(images[i], cmap = 'gray'),\
    plt.title(titles[i]),\
    plt.xticks([]),\
    plt.yticks([])

plt.show()
