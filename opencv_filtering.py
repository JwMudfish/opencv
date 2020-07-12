######### Filtering
# Spatial Filter = (spatial) mask, kernel, window

# Mask 종류
# • 필터링의 역할에 따라 마스크의 형태가 결정됨

# Spatial Filtering 의 역할
# • 영상을 흐리게(blur) 만들기(smoothing)
# • 영상을 날카롭게 만들기 (sharping)
# • 경계선(edge) 검출
# • 노이즈 제거

#########  가장자리 픽셀 처리 방법
# • OpenCV 에서 BorderTypes 를 조정하여 사용 가능

# BorderTypes.Constant 고정 값으로 픽셀을 확장
# BorderTypes.Replicate 테두리 픽셀을 복사해서 확장
# BorderTypes.Reflect 픽셀을 반사해서 확장
# BorderTypes.Wrap 반대쪽 픽셀을 복사해서 확장
# BorderTypes.Reflect101 이중 픽셀을 만들지 않고 반사해서 확장
# BorderTypes.Default Reflect101 방식을 사용
# BorderTypes.Transparent 픽셀을 투명하게 해서 확장
# BorderTypes.Isolated 관심 영역(ROI) 밖은 고려하지 않음

###########  2D filtering 사용방법
'''
cv2.filter2D(src, ddepth, kernel, dst =None, anchor =None, delta =None, borderType =None)
'''
# • src : 입력 영상
# • ddepth
# • 출력 영상 데이터 타입. -1을 지정하면 src 와 같은 타입의 dst 영상 생성
# • (e.g) cv2.CV_8U(numpy 의 uint8), cv2.CV_32F(numpy 의 float32)
# • kernel : 필터 마스크 행렬. 실수형.
# • anchor
# • kernel 내부의 고정점 위치.
# • ( -1, -1 )이면 kernel 중앙으로 사용(3x3 kernel 인 경우 가운데인 (1,1))
# • delta : 추가적으로 더할 값
# • borderTypes : 가장자리 픽셀 확장 방식

#######################################
# [실습]
# numpy 를 사용해서 3x3 median filter 를 이용하여 grayscale 영상을 blurring 해보세요.
'''
import cv2
import numpy as np

src = cv2.imread('./images/apples.jpg', cv2.IMREAD_GRAYSCALE)

# 3x3 filter
kernel = np.array([[1/9, 1/9, 1/9],
                  [1/9, 1/9, 1/9],
                  [1/9, 1/9, 1/9]], dtype=np.float32)

# 5x5 filter
kernel2 = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                  [1/25, 1/25, 1/25, 1/25, 1/25],
                  [1/25, 1/25, 1/25, 1/25, 1/25],
                  [1/25, 1/25, 1/25, 1/25, 1/25],
                  [1/25, 1/25, 1/25, 1/25, 1/25]], dtype=np.float32)

dst = cv2.filter2D(src, -1, kernel)
dst2 = cv2.filter2D(src, -1, kernel2)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''


##############################################################################
###################  Median Filter(평균값 필터)
# • 영상의 좌표 값을 주변 픽셀들의 산술 평균으로 변경
# • 주변값을 이용하여 경계가 흐릿해지는 현상 발생
# • 영상의 잡음이 사라지는 효과
# • mask 크기가 커질수록 결과가 흐릿해짐
# • 띠리서 a 의 오른쪽 부분의 주변이 경계가 모호해짐
# • mask 가 커짐에 따라 convolution 의 계산량 증가
'''
cv2.blur(src, ksize, dst =None, anchor =None, borderType =None)
'''
# • src : 입력 영상
# • ksize : 필터 크기로 (width, height) 형태로 입력
# • dst : 결과 영상, 주로 입력 영상과 같은 크기와 타입을 사용

#######################################
# numpy 로 한 것을 openCV 로 해보세요.

'''
import cv2
import numpy as np

src = cv2.imread('./images/apples.jpg', cv2.IMREAD_GRAYSCALE)

# opencv blur 함수 사용
dst1 = cv2.blur(src, (3,3))

# numpy
kernel = np.ones((3,3), dtype=np.float32) / 9
dst2 = cv2.filter2D(src, -1, kernel)


cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''


##############################################################################
###################  Gaussian filter
# Median Filter 의 특징
# • 주어진 픽셀의 거리에 관계없이 모두 같은 가중치를 반영
#     → 픽셀과의 거리를 고려한 filter 필요
# • Gaussian(Normal Distribution, 정규분포)

# Multivariate Normal Distribution(다변량 정규분포)
#• 이미지는 2차원 array 이 되어있어 2차원 정규분포 필요

'''
cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, dst =None, borderType =None)
'''

# • src : 입력 영상
# • ksize 홀수로 넣어줘야 함
# • Gaussian filter mask 크기
# • (0,0) : sigma 에 의해 자동으로 결정
# • sigmaX , sigmaY : 표준편차
# • bordertypes : 가장자리 픽셀 확장 방식
# 표준편차만 넣어주고 커널은 안쓰는게 좋을 수 있다.

'''
import cv2
import numpy as np

src = cv2.imread('./images/apples.jpg', cv2.IMREAD_GRAYSCALE)

# gaussian blur 함수 사용 - 표준편차를 1,3,5
dst1 = cv2.GaussianBlur(src, (9,9),5,5)

# min filter
dst2 = cv2.blur(src, (9,9))


cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''

##################################################
############  Noise cancel : Median Filter
# • 영상의 잡음(Noise)
# • 우리가 획득 가능한 영상은 원본에 잡음이 섞여있음

# 잡음의 종류
# • Gaussian Noise
# • 소금 & 후추 잡음

# Median filter
# • 커널 내의 픽셀들을 확인하고 순서대로 정렬하여 중간에 해당하는 값을 현재 픽셀에 넣어주는 방식
# • 다른 값이 비해 특별히 크거나 작은 밝기 값을 가지는 소금-후추 노이즈에 아주 효과적

'''
cv2.medianBlur(src, ksize, dst =None)
'''

# • src : 입력 영상(채널별)
# • ksize : kernel 크기, 1보다 큰 홀수
# • dst : 출력 영상, src 와 같은 크기 & 타입
'''
import numpy as np
import cv2

src = cv2.imread('./images/noise2.jpg')

dst1 = cv2.medianBlur(src,3)
dst2 = cv2.medianBlur(src,5)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''

##################################################
############ Bilateral filter(양방향 필터)
# • 미디언 필터는 잡음을 제거하는데 효과가 있지만 경계도 흐릿하게 만듦
# • 가우시안 필터와 경계 필터를 사용하여 노이즈는 없고 경계가 비교적 뚜렷한 영상을 얻을 수 있음
# • 계산량이 많아 속도가 느림

'''
cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
'''
# • src : 입력 영상
# • d : 필터의 직경(diameter), 5 이상이면 속도가 느림
# • -1 이면 sigmaSpace 에 의해 자동 결정됨
# • sigmaColor : 색공간 필터의 표준편차
# • sigmaSpace : 죄표 공간에서 필터의 표준편차
# • borderType : 가장자리 픽셀 처리 방법
# • dst : 출력 영상, src 와 같은 크기 & 타입

import numpy as np
import cv2

src = cv2.imread('./images/noise2.jpg')

dst1 = cv2.bilateralFilter(src,-1, 5, 5)


cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
#cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()