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
'''
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

    if dist < 0.1:
        cv2.rectangle(dst, rc, (0,0,255), 2)

cv2.imshow('obj', obj)
cv2.imshow('src', dst)
cv2.waitKey()
cv2.destroyAllWindows()
'''

###############################################################
# Template Matching(템플릿 매칭)
# • 어떤 물체가 있는 영상에 그 물체가 있는 위치를 찾는 Task

'''
cv2.matchTemplate(src, template, method, result=None, mask=None) -> result
'''
# • src : 입력 영상
# • template : 템플릿 영상( src 보다 크기는 작거나 같아야함)
# • method : 템플릿 비교 방법
#     • result
#     • 비교 결과 행렬
#     • numpy.ndarray
#     • src 의 크기가 (w,h) , template 의 크기가 (w’,h’ ) 이면 result 의 크기는 (w-w’+1,h-h’+1)
'''
import cv2
import numpy as np

src = cv2.imread('./images/ob/template_matching_ex1.png', cv2.IMREAD_GRAYSCALE)
templ = cv2. imread('./images/ob/template_matching_ex2.png', cv2.IMREAD_GRAYSCALE)

res = cv2.matchTemplate(src, templ, cv2.TM_CCOEFF_NORMED)
res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# minMaxLoc 는 minimum 값,  maximun 값, minimum 위치, maximun 위치 
_, maxv, _, maxloc = cv2.minMaxLoc(res)

th, tw = templ.shape[:2]

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0,0,255), 2)

cv2.imshow('res_norm', res_norm)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()
'''

############################################################
# Haar Cascade Classifiers
# • 머신 러닝기반 object detection
# • Rapid Object Detection using a Boosted Cascade of Simple Features, Paul Viola, Michael Jones(2001) 에서 제안됨
# • 직사각형 영역으로 object detection 을 함
# • 찾으려는 object 가 포함된 이미지와 오브젝트가 없는 이미지를 사용하여 Haar Cascade Classifier 를 학습하여 검출함

# • Haar Feature Selection
#     • 이미지를 스캔하면서 위치를 이동시키는 인접한 직사각형들의 영역내에 있는 픽셀의 합의 차이를 이용
#     • 사각 영역 내부의 픽셀들을 빨리 더하기 위해 integral image 를 사용

# • Haar Feature Selection 의 종류 및 적용 방법
#     • 검정색 영역의 합에서 흰색 영역의 합을 뺌
#     • 여러 유형의 feature 들은 edge, line, 중심 대각선을 찾는데 유용함

# • Integral Images 생성 방법
#     • 기존 이미지의 너비와 높이에 1씩 더해서 더 큰 이미지를 만든 후 맨 왼쪽과 맨 위쪽은 0 으로 채움
#     • 결과에 왼쪽 위부터 더한 값을 해당 픽셀값으로 대체
# • 원본영상에서 영역을 지정하여 내부의 값을 구할 때 integral image 에서 대응되는 4 곳의 픽셀값을 이용해서 
#   원본영상의 영역의 합을 구할 수 있음
# • 원본영상의 영역의 크기 = integral image 의 오른쪽 아래 픽셀 – 왼쪽 아래의 왼쪽으로 한 칸 더 왼쪽의 픽셀 + 오른쪽 위에서 한 칸 위의 픽셀 +
#   왼쪽 위에서 한 칸 더 왼쪽 위로 올라간(대각선 방향) 픽셀

# • Cascade Classifier
# • 현재 윈도우가 있는 영역이 얼굴 영역인지를 단계별로 체크하는 방법을 사용
# • 낮은 단계에서는 적은수의 feature 만 사용하여 짧은 시간에 얼굴 영역인지
# 판단하게 되며 상위 단계로 갈수록 좀 더 많은 feature 를 사용하여 시간이 오래걸림(이 방식을 Cascade Classifier 라고 부름)
# • 첫번째 단계의 특징에서 얼굴 영역이 아니라는 판정이 나면 바로 다음 위치로 윈도우를 이동
# • 첫번째 단계의 특징에서 얼굴 영역이라는 판정이 내려지면 현재 윈도우가 위
# 치한 곳에 다음 단계의 특징을 적용

#pretrained XML files
#• https://github.com/opencv/opencv/tree/master/data/haarcascades

'''
cv2.CascadeClassifier.detectMultiScale(image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None,
maxSize=None) -> dst
'''
# • image : 원본 영상
# • scaleFactor : 영상 축소 비율(=검색하는 박스를 늘려가는 비율) , defalut = 1.1
# • minNeighbors : 얼마나 많은 사각형이 근방에서 검출되어야 최종 검출 영역으 로 선택할 것인지를 정하는 값(default = 3)
# • flags : 현재는 미사용
# • minSi ze : 최소 box 크기( (w,h) 형태)
# • maxSi ze : 최대 box 크기( (w,h) 형태)
# • dst : 검출된 객체의 사각형 정보를 담은 np.ndarray, 검출하고자 하는 object가 여러개인 경우 여러개 모두 추출, shape = (n,4) (n : object 갯수)

import cv2
import numpy as np

src = cv2.imread('./images/ob/cascade.jpg')

f_classifier = cv2.CascadeClassifier('./images/ob/haarcascade_frontalface_alt.xml')
eye_classifier = cv2.CascadeClassifier('./images/ob/haarcascade_eye.xml')

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
faces = f_classifier.detectMultiScale(gray, 1.2, 5)


for (x,y,w,h) in faces:
    cv2.rectangle(src, (x,y,w,h), (0,0,255),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = src[y:y+h, x:x+w]

    eyes = eye_classifier.detectMultiScale(roi_gray)

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey),(ex+ew, ey+eh), (0,255,0),2)


cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()

#####################################################################



