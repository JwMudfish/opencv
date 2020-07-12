#### Histogram 함수

# cv2.calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None) -> hist

# • images : 입력 영상 리스트
# • channels : 히스토그래을 그릴 채널 리스트
# • mask : 마스트 영상(전체의 히스토그램을 얻고 싶으면 None)
# • histSize : 히스토그램의 bin 의 갯수
# • ranges : 히스토그램 각 차원의 최솟값과 최대값(리스트 타입)
# • accumulate : 누적 히스토그램을 나타내고 싶으면 True (defalut 는 False)
# • hist : 계산된 히스토그램


##### 흑백 이미지 #######
import matplotlib.pyplot as plt
import cv2
'''
src = cv2.imread('./images/lenna.jpg', cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([src], [0], None, [256], [0,256])
cv2.imshow('src', src)
cv2.waitKey(1)

plt.plot(hist)
plt.show()
'''


#### 컬러 이미지
'''
src = cv2.imread('./images/lenna.jpg')

color = ['b','g','r']

bgr_planes = cv2.split(src)

for (p,c) in zip(bgr_planes, color):
    hist = cv2.calcHist([p], [0], None, [256], [0,256])
    plt.plot(hist, color = c)

cv2.imshow('src', src)
cv2.waitKey(1)

plt.show()
'''

##### 명암비
# • 완전한 흰색과 완전한 검정색의 밝기(휘도) 차이
# • 흰색의 밝기를 검정색의 밝기로 나누어 계산
# • 일반적으로 명암비는 픽셀들에 속하는 영역을 [0, 255] 로 확장함
# 예) 픽셀값이 100 ~ 200 인 grayscale 영상을 0 ~ 255 로 변환

##### Histogram stretching(히스토그램 스트레칭)
# • 영상의 최솟값을 0 , 최댓값을 255 로 영상을 변경
# • 히스토그램이 0 부터 255 까지 고르게 펴지는 효과
# • 소수점 계산으로 인해 중간에 픽셀이 없는 값 존재
# • 0 ~ 255 가 아닌 2 0 ~ 240 같이 새로운 구간 적용 가능

#cv2.normalize(src, dst, alpha=None, beta=None, norm_type=None, dtype=None, mask=None)

# • src : 입력 영상
# • dst : 결과 영상
# • alpha
# • Normed 영상의 최솟값(norm_type = NORM_MINMAX 경우)
# • Normed 영상의 Norm 값(norm_type = NORM_L1/L2 경우)
# • beta : Normed 영상의 최댓값(norm_type = NORM_MINMAX 경우)
# • dtype : 결과 영상의 타입
# • mask : 마스크 영상
'''
src = cv2.imread('./images/dora.jpg', cv2.IMREAD_GRAYSCALE)

dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)

hist1 = cv2.calcHist([src],[0], None, [256], [0, 256])
hist2 = cv2.calcHist([dst],[0], None, [256], [0, 256])

cv2.imshow('src', src)
cv2.imshow('dst', dst)

cv2.waitKey(1)

plt.plot(hist1, label= 'original')
plt.plot(hist2, label= 'normalized')

plt.legend(loc=2)
plt.show()

'''

############## 
#######    Histogram Equalization(히스토그램 평활화)
# • 히스토그램이 영상 전체에서 균일한 분포가 되도록 변경하는 기법
# • 히스토그램 평활화 방법
# 1 . 히스토그램 구하기
# 2 . 히스토그램에서 전체 픽셀수로 나눈 정규화된 히스토그램 구하기
# 3 . 정규화된 히스토그램에서 누적 확률 부포함수 구하기
# 4 . 3의 결과에 기존 영상에서의 최댓값을 곱하기
# 5 . 반올림하기

# dst = cv2.equalizeHist(src)

# 컬러영상의 Histogram Equalization
# • Histogram Equalization 은 흑백 영상만 가능
# • BGR 영상
# • BGR 영상을 B, G, R 각각 평활화하면 영상이 이상함
# • YCrCb 영상
# 1 . 밝기성분(Y) 에 대해서만 평활화
# 2 . Cr, Cb 는 동일하게 유지

'''
src = cv2.imread('./images/sky.jpg')

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
ycrcb_planes = cv2.split(src_ycrcb)

ycrcb_planes[0] = cv2.equalizeHist(ycrcb_planes[0])

dst_ycrcb = cv2.merge(ycrcb_planes)
dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
#cv2.imshow('dst_ycrcb', dst_ycrcb)

cv2.waitKey()
cv2.destroyAllWindows()
'''


#########################
#############   Histogram Backprojection(히스토그램 역투영)
# • 영상에서 특정 색상만 분리하는 방법
# • 2차원 히스토그램과 HSV 또는 YCrCb 색공간을 이용

# cv2.calcBackProject(img, channels, hist, ranges, scale) -> dst

# • img : 입력 영상
# • dst : 결과 영상
# • channel : 처리할 채널(리스트로 표현)
# • 1채널 예시 : [0]
# • 2채널 예시 : [1, 2 ]
# • 3채널 : [0,1,2]
# • hist : 역투영에 사용할 히스토그램
# • ranges : 각 픽셀이 가질 수 있는 값의 범위
# • scale : 역투영 행렬에 추가적으로 곱할 값
'''
src = cv2.imread('./images/mans.jpg')

#src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

x, y, w, h = cv2.selectROI(src)
#crop = src_hsv[y:y+h, x:x+w]
crop = src_ycrcb[y:y+h, x:x+w]


#channels = [0,1]  # hsv
channels = [1,2]  # ycrcb

bin1 = 128
bin2 = 128

histSize = [bin1, bin2]

#h_range = [0,256]
#s_range = [0,256]

cr_range = [0,256]
cb_range = [0,256]


#ranges = h_range + s_range
ranges = cr_range + cb_range


hist = cv2.calcHist([crop], channels, None, histSize, ranges)
#hist_norm = cv2.normalize(cv2.log(hist+1), None, 0, 255, cv2.NORM_MINMAX)


#backproj = cv2.calcBackProject([src_hsv], channels, hist, ranges, 1)
backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

dst = cv2.copyTo(src, backproj)

cv2.imshow('backproj', backproj)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
'''

############################
############   특정 색상 추출
# • 영상 내부의 특정 색상만 검출이 가능
# • BGR, HSV, YCrCb 를 사용해서 각 요소별로 크기를 지정하여 특정 범위의 컬러만 추출
# • 일반적으로 BGR 보다는 HSV 가 성능이 좋음

# cv2.inRange(src, lowerb, upperb, dst=None) -> dst

# • s rc : 입력 영상
# • lowerb : 하한값
# • upperb : 상한값
# • dst
# • 범위 안에 들어가는 픽셀은 255, 나머지를 0 으로 변경
# • 크기는 입력 영상과 동일

'''
src = cv2.imread('./images/apples.jpg')
src = cv2.resize(src, (0,0), fx=0.5, fy= 0.5, interpolation=cv2.INTER_NEAREST)
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

x, y, w, h = cv2.selectROI(src)
crop = src_hsv[y:y+h, x:x+w]

channels = [0,1]  # hsv

bin1 = 128
bin2 = 128

histSize = [bin1, bin2]

h_range = [0,256]
s_range = [0,256]

ranges = h_range + s_range

hist = cv2.calcHist([crop], channels, None, histSize, ranges)
#hist_norm = cv2.normalize(cv2.log(hist+1), None, 0, 255, cv2.NORM_MINMAX)

backproj = cv2.calcBackProject([src_hsv], channels, hist, ranges, 1)

dst3 = cv2.copyTo(src, backproj)


dst1 = cv2.inRange(src, (0,128,0), (100,255,100))  # b 0~100, g 128 ~ 255, r 0 ~ 100
dst2 = cv2.inRange(src_hsv, (50,150,0), (80,255,255))  # h 0~100, s 128 ~ 255, v 0 ~ 100


cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
#cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()
'''

#############################################
##### 크로마키 합성
# 녹색 배경의 물체를 다른 영상에 합성하는 방법
# 1 . inRange 함수로 mask 영상을 만들자
# 2 . bitwise 연산으로 mask 영상과 다른 영상을 합성

def fn(x):
    pass


'''
cv2.namedWindow('dst2',cv2.WINDOW_NORMAL)


cv2.createTrackbar('R','dst2', 0, 255, fn)
cv2.createTrackbar('G','dst2', 0, 255, fn)
cv2.createTrackbar('B','dst2', 0, 255, fn)

cv2.createTrackbar('r','dst2', 0, 255, fn)
cv2.createTrackbar('g','dst2', 0, 255, fn)
cv2.createTrackbar('b','dst2', 0, 255, fn)

while True: 
    cv2.imshow('dst2', dst2) 
    R = cv2.getTrackbarPos('R', 'dst2') 
    G = cv2.getTrackbarPos('G', 'dst2') 
    B = cv2.getTrackbarPos('B', 'dst2') 
    
    r = cv2.getTrackbarPos('r', 'dst2')
    g = cv2.getTrackbarPos('g', 'dst2')
    b = cv2.getTrackbarPos('b', 'dst2')
 
    dst2 = cv2.inRange(src_hsv, (b,g,r), (B,G,R))
    
    k = cv2.waitKey(1) 
    
    if k == 27: 
        break 
    
'''
import numpy as np

src = cv2.imread('./images/man.jpg')
src1 = cv2.imread('./images/street.jpg')

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

height1, width1 = src.shape[:2]
height2, width2 = src1.shape[:2]

x = (width2 - width1) // 2
y = height2 - height1
w = (width2 + width1)//2
h = height2

lower = np.array([50, 100, 100])
upper = np.array([80, 255, 255])


#dst2 = cv2.inRange(src, (0,0,0), (255,255,255))  # (29,127,6), (109,255,73)
#mask = cv2.inRange(src_hsv, (24,148,110), (255,255,255))  # (24,148,110), (255,255,255)
mask = cv2.inRange(src_hsv, lower, upper)  # (24,148,110), (255,255,255)

mask_inv = cv2.bitwise_not(mask)

roi = src1[y:h, x:w]
fg = cv2.bitwise_and(src, src, mask = mask_inv)
bg = cv2.bitwise_and(roi, roi, mask = mask)
src1[y:h, x:w] = fg + bg


cv2.imshow('mask', fg)
cv2.waitKey()
cv2.destroyAllWindows()
