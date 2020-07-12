### 영상의 밝기 조절

# 덧셈연산
#cv2.add(src1, src2, dst=None, mask=None, dtype=None)

# 밝기 조절을 위한 영상의 덧셈
# • src1 : 첫 번째 입력 영상
# • src2 : 두 번째 입력 영상
# • dst : 출력 영상 결과
# • mask : 마스크 영상
# • dtype : 출력 영상(dst) 의 데이터 타입
# 예) cv2.CV_8U, cv2.CV_32F 등

import cv2
import numpy as np
import matplotlib.pyplot as plt

#src = cv2.imread('./images/lenna.jpg', cv2.IMREAD_GRAYSCALE)
src = cv2.imread('./images/lenna.jpg')



dst_add = cv2.add(src, 50)



# 뺄셈연산
# cv2.subtract(src1, src2, dst=None, mask=None, dtype=None)
# • src1 : 첫 번째 입력 영상
# • src2 : 두 번째 입력 영상
# • dst : 출력 영상 결과
# • mask : 마스크 영상
# • dtype : 출력영상(dst) 타입


####### 영상끼리 산술연산####################################
### Weighted sum(알파 블렌딩)
#          dst( x,y) = t x src1( x,y) + (1- t) x src2( x,y)
# • 두 개의 입력 src1, src2 의 weight 의 합을 1로 맞추면 두 영상을 유지
# • 합을 1이 되지 않게 하면, 밝기를 잃게 됨

# cv2.addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None)
# • src1 : 첫 번째 입력 영상
# • src2 : 두 번째 입력 영상
# • alpha : 첫 번째 영상 가중치
# • beta : 두 번째 영상 가중치
# • gamma : 영상에 공통적으로 더하는 값
# • dst : 출력 영상 결과
# • dtype : 출력영상(dst) 타입



### Average
# • Weighted sum 의 특별한 경우(t = 0.5)
#         dst( x,y) = 1/2 x src1( x,y) + 1/2 x src2( x,y)


# 차이(절대값) 연산
# dst(x,y) = |src1(x,y) - src2(x,y) |
# • 두 영상의 같은 위치에 있는 픽셀값의 차이를 출력
# • 연속된 영상의 경우 변화가 있는 물체의 결과만 출력

# cv2.absdiff(src1, src2, dst=None)
# • src1 : 첫 번째 입력 영상
# • src2 : 두 번째 입력 영상
# • dst : 출력 영상 결과

####################################################################################

# <실습>
# lenna 영상과 src2 영상(직접 만들기) 를 활용하여 두 영상의 add,
# substract, weighted sum, abs diff 결과를 하나의 창(subplot)에 나타내는 프로그램을 작성하세요(총 6장)

src1 = cv2.imread('./images/lenna.jpg', cv2.IMREAD_GRAYSCALE)
#src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2RGB)

#print(src1.shape[:2])
h, w = src1.shape
src2 = np.zeros((h,w), dtype=np.uint8)
cv2.rectangle(src2, (35,35), (115,115), (255,255,255), -1)


dst1 = cv2.add(src1, src2, dtype = cv2.CV_8U)
dst2 = cv2.subtract(src1, src2, dtype=cv2.CV_8U)
dst3 = cv2.addWeighted(src1,0.5, src2, 0.5, 0) 
dst4 = cv2.absdiff(src1, src2)


#cv2.imshow('src1', src1)
#cv2.imshow('src2', src2)

#cv2.imshow('dst1', dst1)
#cv2.imshow('dst2', dst2)
#cv2.imshow('dst3', dst3)
#cv2.imshow('dst4', dst4)

plt.subplot(231)
plt.title('src1')
plt.axis('off')
plt.imshow(src1, cmap='gray')

plt.subplot(232)
plt.title('src2')
plt.axis('off')
plt.imshow(src2, cmap='gray')

plt.subplot(233)
plt.title('dst1')
plt.axis('off')
plt.imshow(dst1, cmap='gray')

plt.subplot(234)
plt.title('dst2')
plt.axis('off')
plt.imshow(dst2, cmap='gray')

plt.subplot(235)
plt.title('dst3')
plt.axis('off')
plt.imshow(dst3, cmap='gray')

plt.subplot(236)
plt.title('dst4')
plt.axis('off')
plt.imshow(dst4, cmap='gray')

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()

