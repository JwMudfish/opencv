# Gaussian Mixture Model Background
# • Mixture of Gaussian(=Gaussian Mixture Model, GMM) 을 사용하여 픽셀별로 background 와 foreground 를 구분하는 모델

######### BackgroundSubtractorMOG2 클래스 생성 함수
'''
cv2.createBackgroundSubtractorMOG2(, history=None, varThreshold=None, detectShadows=None) -> dst
'''
# • history : 사용할 frame 갯수 (defalut = 500)
# • varThreshold
#     • 픽셀과 모델 사이의 Mahalanobis distance 거리의 제곱에 대한 threshold
#     • 해당 픽셀이 배경 모델에 의해 잘 표현되는지를 판단
#     • default = 16
# • detectShadows : 그림자 검출 여부(default = True )


########## Foreground 객체 마스크 생성 함수
'''
cv2.BackgroundSubtractor.apply(img, fgmask=None, learningRate=None) -> fgmask
'''
# • img : 입력 영상
# • fgmask : 출력 영상
#     • foreground 마스크 영상
#     • 0 (배경) , 128(그림자) , 255(전경) 세 가지 값을 가짐
# • learningRate
#     • 배경 모델 학습 속도
#     • 0 ~ 1 사이의 값 지정
#     • 음수 입력시 자동으로 모델에서 learning rate 지정(default = 1)


############ Background 영상 반환 함수
'''
cv2.BackgroundSubtractor.getBackgroundImage(, backgroundImage=None) -> backgroundImage
'''
# • backgroundImage : 학습된 배경 영상을 출력


import cv2
import numpy as np

cap = cv2.VideoCapture('./images/ob/PETS2000.avi')

###
bs = cv2.createBackgroundSubtractorMOG2()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = bs.apply(gray)
    back = bs.getBackgroundImage()

    cnt, _, stats, _ = cv2.connectedComponentsWithStats(fgmask)

    for i in range(1,cnt):
        x, y, w, h, s = stats[i]

        if s < 50:
            continue

        cv2.rectangle(frame, (x,y,w,h), (0,0,255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('back', back)
    cv2.imshow('fgmask', fgmask)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()