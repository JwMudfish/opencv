######## Mean Shift Algorithm(평균 이동 알고리즘)
# • Tracking Algorithm
# • 데이터가 가장 밀집한 부분을 찾는 알고리즘
# • 좌표상에서 원을 그려서 원 내부에 들어있는 점들의 평균을 구해서 평균으로 원의 중심을 이동하는 방식으로 학습됨
# https://m.blog.naver.com/PostView.nhn?blogId=msnayana&logNo=80109766471&proxyReferer=https:%2F%2Fwww.google.com%2F

'''
cv2.meanShift(img, window, criteria) -> retval, window
'''

# • img : 히스토그램 역투영 영상
# • window : 포기 검색 영역 window, 결과 영역을 ( x,y,w,h) 로 반환
# • criteria
#   • 알고리즘 종료 기준으로 (type, 최대 반복 횟수, 정확도) 형태의 튜플
#   • ex) (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) 
#          최대 반복횟수가 10, 정확도가 1 이하이면 종료
# • retval : 알고리즘 반복 횟수

import numpy as np
import cv2

cap = cv2.VideoCapture('./images/ob/slow.flv')

ret, frame = cap.read()

x,y,w,h = cv2.selectROI('select window', frame)

rc = (x,y,w,h)

roi = frame[y:y+h, x:x+w]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

channels = [0,1]
ranges = [0,180, 0, 256]
hist = cv2.calcHist([roi_hsv], channels, None, [90,128], ranges)

while True:
    ret, frame = cap.read()

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([frame_hsv], channels, hist, ranges, 1)

    _, rc = cv2.meanShift(backproj, rc, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

    cv2.rectangle(frame, rc, (0,0,255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(60) == 27:
        break

cap.relese()
cv2.destroyAllWindows()

