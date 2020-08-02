######### Moving Average Backgroud(이동 평균 배경)
# • 일반적으로 과거 영상에 더 높은 weight 를 부여
# • alpha = 0.01
'''
cv2.accumulateWeighted(img, dst, alpha, mask=None) -> dst
'''
# • img : 입력 영상
# • dst : 결과 영상
# • alpha : 입력 영상(frame) 의 가중치(weight)
# • mask : 마스크 영상


import cv2
import numpy as np

cap = cv2.VideoCapture('./images/ob/PETS2000.avi')
_, back = cap.read()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0,0), 1.0)

##
fback = back.astype(np.float32)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 1.0)

    ###
    cv2.accumulateWeighted(gray, fback, 0.01)
    back = fback.astype(np.uint8)

    diff = cv2.absdiff(gray, back)

    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)


    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)  # 레이블링

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 100:
            continue

        cv2.rectangle(frame, (x,y,w,h), (0,0,255), 2)

    cv2.imshow('back', back)
    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()