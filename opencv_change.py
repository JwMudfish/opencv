############# 영상의 기하학적 변환

# 영상의 기하학적 변환
# • 영상을 이동, 확대/축소, 회전, 뒤틀기 등을 활용하여 원본 이미지를 변환
# • 선형변환의 경우 행렬 곱으로 표현 가능

# 이동(Transla tion )
# • 축의 방향대로 이동하는 변환
# • 각각의 축에 대해 이동하여 합성하는 변환으로 표현 가능

################# 영상의 이동/변환
'''
cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None) -> dst
'''
# • src : 입력 영상
# • M : 2 x 3 변환 행렬
# • dsize
# • 결과 영상 크기(w, h) 형태의 튜플
# • (0,0) 이면 입력영상과 동일한 크기
# • flags : 보간법(defalut 는 선형보간법 cv2.INTER_LINEAR)
# • borderMode : 가장자리 픽셀 확장 방식(defalut 는 cv2.BORDER_CONSTANT )
# • borderValue
# • borderMode 가 cv2.BORDER_CONSTANT 일 때 사용하는 값.
# • defalut 는 0
'''
import numpy as np
import cv2

src = cv2.imread('./images/fish.jpg')

aff = np.array([[1,0,20],
                [0,1,10]], dtype=np.float32)

dst = cv2.warpAffine(src, aff, (0,0))
dst2 = cv2.warpAffine(src, aff, (0,0), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, (255,0,0))


cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''

##############################################
##############  영상의 확대/축소 변환
# • 축소시 고려 사항
# • 축소시 흐릿하게 될 수 있음(예리한 테두리가 없어질 수 있음)
# • cv2.resize 함수에서 cv2.INTER_AREA flag 사용

'''
cv2.resize(src, dsize, dst=None, fx=None, fy=None, interpolation=None) -> dst
'''
# • src : 입력 영상
# • dsize
# • 결과 영상 크기(w, h) 형태의 튜플
# • (0,0) 이면 f x , f y 으로 결정
# • f x , f y : x, y 방향의 변환 비율(dsize = (0,0) 일때 사용)
# • interpolation : 보간법(default 는 cv2.INTER_LINREA)
# cv2.INTER_NEAREST 최근방 이웃 보간법
# cv2.INTER_LINEAR 양선형 보간법 (2x2 이웃 픽셀 참조)
# cv2.INTER_CUBIC 3차회선 보간법 (4x4 이웃 픽셀 참조)
# cv2.INTER_AREA 영상축소시효과적


'''
import numpy as np
import cv2

src = cv2.imread('./images/fish.jpg')

dst1 = cv2.resize(src, (0,0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
dst2 = cv2.resize(src, (512,512))
dst3 = cv2.resize(src, (512, 512), interpolation=cv2.INTER_CUBIC)
dst4 = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.imshow('dst3', dst3)
cv2.imshow('dst4', dst4)

cv2.waitKey()
cv2.destroyAllWindows()

'''
######################################################################################################
#################  영상의 전단 변환
# • 전단(Shear)
# • x 축 (또는 y 축) 만 k 배 비율 변환
'''
import numpy as np
import cv2

src = cv2.imread('./images/fish.jpg')

# 전단 변환행렬 만들기

k=1

aff = np.array([[1, 0, 0],
                [k, 1, 0]], dtype=np.float32)

h, w = src.shape[:2]

#dst1 = cv2.warpAffine(src, aff, (0,0))
dst1 = cv2.warpAffine(src, aff, (w, h + int(k * w)))

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)


cv2.waitKey()
cv2.destroyAllWindows()
'''

#################  영상의 대칭 변환
# • 대칭(Reflection)
# • 좌/우 대칭
# • 상/하 대칭
# • 좌/우 & 상/하 대칭
'''
cv2.flip(src, flipCode, dst=None) -> dst
'''
# • src : 입력 영상
# • flipCode : 대칭 방향
# • 1 : 좌/우 대칭
# • 0 : 상/하 대칭
# • -1 : 좌/우 & 상/하 대칭
'''
import numpy as np
import cv2

src = cv2.imread('./images/fish.jpg')

# 전단 변환행렬 만들기


dst1 = cv2.flip(src, -1)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
# cv2.imshow('dst2', dst2)
# cv2.imshow('dst3', dst3)
# cv2.imshow('dst4', dst4)

cv2.waitKey()
cv2.destroyAllWindows()
'''

###################################################################################
##################  이미지 피라미드
# • 원본 영상을 다양한 크기로 resizing 하는 Ta s k
# • 가우시안 블러링 & 다운샘플링을 사용하여 축소

#이미지 피라미드 다운샘플링
'''
cv2.pyrDown(src, dst=None, dstsize=None, borderType=None) -> dst
'''
# • src : 입력 영상
# • dst : 출력 영상
# • dstsize : 출력 영상 크기(defalut 는 기존 영상의 가로/세로가 각각 ½배
# • borderType : 가장자리 픽셀 확장 방식
# 원리
# • 5x5 Gaussian filter 적용
# • 짝수번째 행/열을 제거


#이미지 피라미드 업샘플링
'''
cv2.pyrUp(src, dst=None, dstsize=None, borderType=None) -> dst
'''
# • src : 입력 영상
# • dst : 출력 영상
# • dstsize : 출력 영상 크기(defalut 는 기존 영상의 가로/세로가 각각 2배
# • 원리
# • 짝수번째 행/열에 픽셀 추가
'''
import numpy as np
import cv2

src = cv2.imread('./images/lj.jpg')

# 직사각형 위치
rc = (380, 400, 300, 300)

#cpy = src.copy()

#cv2.rectangle(cpy, rc, (0,0,255), 2)

for i in range(1,4):
    src = cv2.pyrDown(src)
    cpy = src.copy()
    cv2.rectangle(cpy, rc, (0,0,255), 2, shift= i)
    cv2.imshow('src', cpy)
    cv2.waitKey()


cv2.destroyAllWindows()
'''

############################################################
##############  영상의 회전 변환
# 회전(Roration)
# • 원본 영상을 특정 좌표를 기준으로 회전하는 변환

'''
cv2.getRotationMatrix2D(center, angle, scale) -> retval
'''
# • center : 회전의 중심 좌표로 ( x,y) 튜플
# • angle : 회전하는 각도(반시계방향이 양수)
# • scale : 추가적인 확대 비율
# • retval : 2x3 어파인 변환 행렬

# • cv2.getRotationMatrix2D 의 결과는 2x3 어파인 변환 행렬이므로 결과를 cv2.warpAffine 함수의 parameter 에 넣어야함
# 영상의 중심을 기준으로 45도 만큼 회전(크기 변화는 없음)
'''
rot1 = cv2.getRotationMatrix2D(cp, 45, 1)
'''
# 영상의 중심을 기준으로 90도 만큼 회전(1/2 크기로 변화)
'''
dst1 = cv2.warpAffine((100,100), rot1, (0, 0))
'''

import numpy as np
import cv2

src = cv2.imread('./images/fish.jpg')

cp = (src.shape[1] / 2, src.shape[0] / 2)  # center point

rot1 = cv2.getRotationMatrix2D(cp, 45, 1)
rot2 = cv2.getRotationMatrix2D(cp, 90, 0.5)

dst1 = cv2.warpAffine(src, rot1, (0,0))
dst2 = cv2.warpAffine(src, rot2, (0,0))

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()

