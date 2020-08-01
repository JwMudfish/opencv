# HOG(Histogram of Oriented Gradient)
# • 보행자 검출을 목적으로 만들어진 Descriptor
# • 유용한 정보를 추출하고 관계없는 정보를 버림으로써 이미지를 단순화하는 Feature Descriptor 의 한 종류
# • 영상에서 object 가 있는 윈도우를 찾아 추출 후 64x128x3 크기로 Resizing

# • Sobel 필터를 이용하여 Gradient 크기와 방향을 찾음
# • 계산된 Gradient 의 방향과 크기를 이용하여 9개 bin 으로 구성된 히스토그램으로 변환
# • 9개의 bin 은 각도를 기준으로 0~20, 2 0~ 40 , …, 160~0
# • 정확한 값이 아닌 경우 bin 의 양쪽에 각도를 기준으로 크기를 비율로 계산하여 배분

# • Sobel filter 를 이용하여 Gradient 를 구한다.
# • Gradient 의 크기와 방향을 구한다.
# • 방향은 360도를 총 9개로 나눈다
# • 반대 방향(예를 들어 0 도와 180 도는 동일하게 본다)
# • 하나의 8x8 블록 당 9개의 벡터 방향이 있고, 8x8 형태의 kernel 이 2x2
#   형태로 되어 있으므로, 9x2x2 = 36 가지 경우의 수가 생김
# • 8x8 형태를 2x2 형태로 묶은 것을 다시 stride 8 로 이동하므로 가로로 7
#   번, 세로로 15번 갈 수 있어서 결과적으로 36x7x15 = 3780 크기가 됨


# HOG detector 객체
'''
cv2.HOGDescriptor()
'''

# Pretrained 된 feature vectors
'''
cv2.HOGDescriptor_getDefaultPeopleDetector() -> retval
'''
# • retval : Pretrained 된 feature vectors


# SVM Classifier
'''
cv2.HOGDescriptor.setSVMDetector(svmdetector) -> None
'''
# • svmdetector : SVM 의 input data

'''
cv2.HOGDescriptor.detectMultiScale(img, hitThreshold=None, winStride=None, padding=None, scale=None,
finalThreshold=None, useMeanshiftGrouping=None) -> rects, weights
'''
# • img : 입력 영상
# • scale : 검색 윈도우 크기 확대 비율(default = 1.05)
# • rects : 검출된 결과 영역 좌표 (n x 4(x, y, w, h) 형태)
# • weights : 검출된 결과 계수 n x 1

import cv2

cap = cv2.VideoCapture('./images/ob/walking.avi')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()

    if not ret:
        break

    detected, _ = hog.detectMultiScale(frame)

    for (x,y,w,h) in detected:
        cv2.rectangle(frame, (x,y,w,h), (0,255,0), 3)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()





