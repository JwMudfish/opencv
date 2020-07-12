## 영상생성 및 부분 영상 추출
import cv2
import numpy as np

'''
# grayscale
img1 = np.empty((480, 640), dtype=np.uint8)


# 컬러 but 블랙
img2 = np.zeros((480, 640, 3), dtype=np.uint8) # 컬러 영상 & zeros 이므로 0 이어서 검정색

# 컬러, white
img3 = np.ones((480, 640), dtype=np.uint8) * 255 # white (1 에 255 를 곱해서 모든 원소가 255)


img4 = np.full((480, 640, 3), (0, 255, 255), dtype=np.uint8) # yellow
img5 = np. full((480, 640), 128, dtype=np.uint8) # grayscale & gray 색상


'''

'''
## 부분 영상 추출
img1 = cv2.imread('./images/apple.jpg',cv2.IMREAD_COLOR)
print(img1.shape)

img2 = img1[50:100, 50:100]  # 가로, 세로 50 ~100을 복사
img3 = img1[50:100, 50:100].copy()  # 가로, 세로 50 ~100을 복사

img2.fill(0)  # img2 를 검정색으로 변경

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
#cv2.imshow('img4', img4)
#cv2.imshow('img5', img5)

'''
##############################################################
# 영상의 속성 (그리기)
# 그리기 알고리즘으로 픽셀값이 변경됨
# 원본 영상이 변경되므로 재사용 필요시 copy 필요
# grayscale 영상에는 그리기 안 됨
# cv2.cvtColor() 함수로 BGR 로 변경 후 그리기 사용

############################################################
## 직선 그리기
# img : 그림을 그릴 원본 이미지
# pt1, pt2 : 시작점, 끝점
# color : 선 색상 또는 밝기, (B,G,R)튜플또는정수값
# thickness : 선두께, 기본값 1
# lineType : 선타입
# cv2.LINE_4, cv2.LINE_8(default), cv2.LINE_AA
# shift : 그리기 좌표값의 축소비율. 기본값 0
# cv2.line(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)

# 도화지 세팅
# img = np.full((480, 640,3),(255,255,255), dtype=np.uint8)

# 선 그리기
#cv2.line(img, (100,100), (300,300), (0,0, 255), 10)
#cv2.line(img, (200,100), (400,300), (0,0, 255), 10)
#cv2.line(img, (300,100), (500,300), (0,0, 255), 10)

###########################################################
# 사각형 그리기
# img : 그림을 그릴 원본 이미지
# pt1, pt2 : 사각형의 왼쪽 끝, 오른쪽 끝
# rec : 사각형 위치, ( x,y,w,h) 튜플
# color : 선 색상 또는 밝기, (B,G,R) 튜플 또는 정수
# thickness : 선두께, 기본값 1 , - 1 은 내부를 채움
# lineType : 선타입
# cv2.LINE_ 4, cv2.LINE_ 8(default), cv2.LINE_AA
# shift : 그리기 좌표값의 축소비율. 기본값 0
# cv2.rectangle(img, pt1, pt2, color, thickness=None, lineType=None, shift=None)
# cv2.rectangle(img, rec, color, thickness=None, lineType=None, shift=None)

#cv2.rectangle(img, (50,50), (300,400), (255,0,0))
#cv2.rectangle(img, (150,150), (400,500), (0,255,0), 5)


# 다각형 그리기
'''
pts1 = np.array([[50,50], [150,150], [100,140], [200,240]], dtype=np.int32)
pts2 = np.array([[350,50],[250,200],[450,200]], dtype=np.int32)


cv2.polylines(img, [pts1], True, (255,0,0))
cv2.polylines(img, [pts2], True, (0,255,0), thickness=5)
'''

#########################################################
# 문자열 출력
# img : 그림을 그릴 원본 이미지
# text : 출력할 문자열
# org : 영상에서 문자열을 출력할 위치의 좌측 하단 좌표, (x, y) 튜플
# fontFace : 폰트 종류. cv2.FONT_HERSHEY _ 로 시작하는 상수 중 선택
# color : 선 색상 또는 밝기, (B,G,R)튜플또는정수값
# thickness : 선두께, 기본값 1
# lineType : 선타입, cv2.LINE_ 4, cv2.LINE_ 8(default), cv2.LINE_AA
# bottomLeftOrigin: True는 영상의 좌측 하단을 원점으로 간주. 기본값은 False.
# cv2.putText(img, text, org, fontFace, fontScale, color, thickness=None, lineType=None, bottomLeftOrigin=None)

img_w = 1280
img_h = 1080

img = np.zeros((img_w, img_h,3), dtype=np.uint8)

center_x = int(img_w / 2)
center_y = int(img_h / 2)

cv2.putText(img, 'FONT_HERSHEY_SCRIPT_SIMPLEX',(center_x - 500, center_y - 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (255,0,0))




cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()