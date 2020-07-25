import cv2
import numpy as np
import math

############# 외곽선 검출 ########
'''
cv2.findContours(src, mode, method, contours=None, hierarchy=None, offset=None) -> contours, hierarchy
'''
# • src : 입력 영상(0 이 아니면 물체로 인식)
# • mode : 외곽선 검출 방식
# • method : 외곽선 근사 방법
# • contours
#     • 외곽선 좌표(np.ndarray)
#     • len(contours) : 외곽선 갯수
# • hierarchy
#     • 외곽선 계층 정보를 담고 있는 (1,n,4) shape 의 list(n 은 contour 갯수)
#     • next, previous, child, parent 를 의미
# • offset : 좌표 이동 offset(default = (0,0))    

# • 외곽선 검출 방식(mode)
# • cv2.RETR_EXTERNAL
#     • 가장 바깥쪽의 영역만 추출
#     • 계층 정보 없음
# • cv2.RETR_LIST
#     • 모든 영역을 추출
#     • 계층 정보 없음
# • cv2.RETR_TREE
#     • 바깥 영역부터 계층적 구조를 추출

######################################
##### 외곽선 길이
'''
cv2.arcLength(curve, closed) -> retval
'''
# • curve : 외곽선 좌표, numpy.ndarray 타입, shape = (n, 1, 2)
#     • 예) findContours 결과로 얻은 contours
# • closed : True 이면 시작점과 끝을 이어서 closed curve(폐곡선)으로 간주 - 시작과 끝점이 만나는..
# • retval : 외곽선 길이

##### 면적
'''
cv2.contourArea(curve, oriented = None) -> retval
'''
# • curve : 외곽선 좌표, numpy.ndarray 타입, shape = (n, 1, 2)
#     • 예) findContours 결과로 얻은 contours
# • oriented
#     • True : 외곽선 진행 방향이 시계방향/반시계방향에 따라 + / - 가 변환
#     • False : 벙향에 관계없이 +
#     • default = False
# • retval : 외곽선으로 둘러 쌓인 영역의 면적


####### Bounding Box, Circle
'''
cv2.boundingRect(curve) -> retval
'''
# • curve : 외곽선 좌표, numpy.ndarray 타입, shape = (n, 1, 2)
#     • 예) findContours 결과로 얻은 contours
# • retval : 사각형 정보 (x, y, w, h)

'''
cv2. minEnclosingCircle(curve) -> center, radius
'''
# • curve : 외곽선 좌표, numpy.ndarray 타입, shape = (n, 1, 2)
    # • 예) findContours 결과로 얻은 contours
# • center : bounding circle 의 중심 좌표, ( x,y) 튜플
# • radius : bounding circle 의 반지름, 실수값

'''
img = cv2.imread('./images/arc_length.png', cv2.IMREAD_GRAYSCALE)

# 외곽선 따기 위해서 이진화 먼저
# 컨투어 구할때 바이너리인버스 해서 구하는게 좋다!!!!
ret, thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 가장 바깥쪽 컨투어만

#print(contours)
#print(len(contours))

print('contour의 길이 : ', cv2.arcLength(contours[0], True))
print('contour 포함하는 rectangle : ', cv2.boundingRect(contours[0]))
print('contour 내부영역 넓이 : ', cv2.contourArea(contours[0]))
print('contour를 둘러싸는 bounding circle : ', cv2.minEnclosingCircle(contours[0]))


cv2.imshow('img', img)
cv2.imshow('thr', thr)
cv2.waitKey()
cv2.destroyAllWindows()
'''

######################################################################
######### Douglas–Peucker algorithm

# • 양 끝점을 기준으로 미리 정한 일정 거리(Threshold) 보다 멀어진 점은 살리고 일정 거리 미만의 점은 제외하면서 단순화하는 알고리즘
# • 양 끝점을 잇는 선분과 선분에서 가장 멀리 떨어진 점 A 가 Threshold 보다멀리 떨어져 있는지 확인하여 Threshold 보다 멀리 떨어져 있으면 남기고,
#   Threshold 보다 가까우면 제거한다.
# • 가장 멀리 떨어진 점 A 와 양 끝점을 잇는 선분 두 개에 대해 마찬가지로 선분과 가장 멀리 떨어진 점이 Threshold 보다 멀리 떨어져 있으면 남기고, 아니면 제거한다.
# • Threshold 는 외곽선의 길이의 1 ~ 5% 범위에서 사용하며 cv2.a rcLeng th (p ts, True ) * 0.01 ~ 0.05 로 적용 가능하다.

'''
cv2.approxPolyDP(curve, epsilon, closed, approxCurve=None) -> approxCurve
'''
# • curve : 외곽선 좌표, numpy.ndarray 타입, shape = (n, 1, 2)
#       예) findContours 결과로 얻은 contours
# • epsilon
# •     근사화 정밀도(curve 와 근사화 곡산 간의 최대 거리)
# •     cv2.a rcLeng th (p ts, True ) * 0.01 ~ 0.05 로 많이 사용됨
# • closed : True 이면 시작점과 끝을 이어서 closed curve(폐곡선)으로 간주
# • approxCurve : 근사화된 곡선의 좌표, numpy.ndarray 타입, shape = (n, 1, 2)

####### • Convex set(컨벡스 집합)
# 집합 내 임의의 두 점에 대해 두 점을 잇는 선분이 집합 내부에 있는 집합

'''
cv2.isContourConvex(contour) -> retval
'''
# • curve : 외곽선 좌표, numpy.ndarray 타입, shape = (n, 1, 2)
#      • 예) findContours 결과로 얻은 contours
# • retval : convex 이면 True, 아니면 False

# • 도형 검출
# • 다각형 : Douglas–Peucker algorithm 을 사용하여 꼭지점 갯수로 확인
# • 원 : 원의 경우 원주와 넓이를 이용하여 확인, 따라서 1 근처의 값일 경우 원으로 판단


#############################################################
#### 영상 내의 도형을 검출해보세요 ##########
'''
def setLabel(img, pts, label):
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x,y)
    pt2 = (x + w, y + h)
    cv2.rectangle(img, pt1, pt2, (0,0,255), 1)
    cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))


img = cv2.imread('./images/polygon.png')


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) 

contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 가장 바깥쪽 컨투어만

for pts in contours:
    if cv2.contourArea(pts) < 500:  # 너무 작으면 무시
        continue

    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.01, True)

    vtc = len(approx)

    if vtc == 3:
        setLabel(img, pts, 'TRI')
    elif vtc == 4:
        setLabel(img, pts, 'RECT')
    elif vtc == 5:
        setLabel(img, pts, '5GON')
    elif vtc == 6:
        setLabel(img, pts, 'PENTAGON')
    elif vtc == 7:
        setLabel(img, pts, 'OCTAGON')
    else:
        lenth = cv2.arcLength(pts, True)
        area = cv2.contourArea(pts)
        ratio = 4. * math.pi * area / (lenth * lenth)

        if ratio > 0.87:
            setLabel(img, pts, 'Circle')


cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

'''

################################################################
#### 영상내에 있는 문서를 자동으로 검출하는 프로그램을 작성해보세요.

def reorderPts

src = cv2.imread('./images/namecard/22.jpg')

# 출력영상 설정
dw, dh = 720, 480
srcQuad = np.array([[0,0], [0,0],[0,0],[0,0]], np.float)


src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# 외곽선 근사화 
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.01, True)

# 컨백스가 아니거나, 사각형이 아니면 무시
    if not cv2.isContourConvex(approx) or len(approx) != 4:
        continue

    cv2.polylines(cpy, )





cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()

