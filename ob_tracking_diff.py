#########  Stationary background subtraction(정적 배경 차분)
# • 기존의 배경과 현재 입력 프레임과의 차이를 이용하여 움직이는 물체 검출

import cv2
import numpy as np

cap = cv2.VideoCapture('./images/ob/PETS2000.avi')
_, back = cap.read()

back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
back = cv2.GaussianBlur(back, (0,0), 1.0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0,0), 1.0)

    diff = cv2.absdiff(gray, back)
    _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)


    cnt, _, stats, _ = cv2.connectedComponentsWithStats(diff)  # 레이블링

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 100:
            continue

        cv2.rectangle(frame, (x,y,w,h), (0,0,255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('diff', diff)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()