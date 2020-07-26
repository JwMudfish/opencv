# Moments 기반 Object Detection
# Moments
# • 영상의 모든 픽셀 좌표의 정보를 이용하여 객체의 모양을 기술하는 방법
# • 2차원 영상을 함수 f ( x,y)의 형태로 나타낼 경우, 기하학적 모멘트 (geometric moment)는 다음과 같이 정의됨
# • p+q : Moments 의 차수( p와 q는 0 이상의 정수)
# • M과 N은 각각 영상의 가로와 세로 픽셀의 크기를 나타낸다.

# Moments 종류
# • Geometric Moments(기하학적 모멘트)
# • Central Moments(중심 모멘트)
#     • 기하학적 모멘트는 영상의 이동 변환시 그 값이 크게 변한다는 단점이 있음.
#     • 영상의 무게 중심을 고려하여 모멘트를 계산하는 방법
# • Hu 의 Moments
#   • H u는 3차 이하의 중심 모멘트를 조합하여 만든 7개의 불변 모멘트(invariant moments)를 만듦
#   • 크기를 정규화한 중심 모멘트를 비선형으로 조합하여 만들어짐
#   • 영상의 크기, 회전, 위치에 불변

'''
cv2.matchShapes(contour1, contour2, method, parameter) -> retval
'''
# • contours1 : 첫 번째 contour 또는 grayscale 영상
# • contours2 : 두 번째 contour 또는 grayscale 영상
# • method : 비교 방법
#     • cv2.CONTOURS_MATCH_I1,cv2.CONTOURS_MATCH_I2, cv2.CONTOURS_MATCH_I3
# • parameter : 현재 미사용
# • retval : 두 contours 또는 grayscale 영상 사이의 거리

import numpy as np
import cv2


obj = cv2.imread('./images/ob/spades.png', cv2.IMREAD_GRAYSCALE)
src = cv2.imread('./images/ob/symbols.png', cv2.IMREAD_GRAYSCALE)

_, obj_bin = cv2.threshold(obj, 128, 255, cv2.THRESH_BINARY_INV)
obj_contours, _ = cv2.findContours(obj_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print(len(obj_bin))
obj_pts = obj_contours[0]

_, src_bin = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)
src_contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for pts in src_contours:
    if cv2.contourArea(pts) < 100:
        continue

    rc = cv2.boundingRect(pts)  # x y w h
    cv2.rectangle(dst, rc, (255,0,0), 2)

    dist = cv2.matchShapes(obj_pts, pts, cv2.CONTOURS_MATCH_I3, 0)

    cv2.putText(dst, str(round(dist, 4)), (rc[0], rc[1] -3), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255,0,0), 1, cv2.LINE_AA)



cv2.imshow('obj', obj)
cv2.imshow('src', dst)
cv2.waitKey()
cv2.destroyAllWindows()

