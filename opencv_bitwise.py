#영상의 논리 연산을 사용하여 결과를 확인해보세요
#• 아래 사진을 흑백으로 읽은 후 bitwise 연산을 실행해 보세요

'''
영상의 논리 연산
• 픽셀별로 논리연산 후 이진수로 표현

cv2.bitwise_and(src1, src2, dst=None, mask=None)
cv2.bitwise_or(src1, src2, dst=None, mask=None)
cv2.bitwise_xor(src1, src2, dst=None, mask=None)
cv2.bitwise_not(src1, dst=None, mask=None)

• src1 : 첫 번째 입력 영상
• src2 : 두 번째 입력 영상
• dst : 출력 영상 결과
• mask : 마스크

'''

import cv2
import numpy as np

src1 = cv2.imread('./images/star.jpg', cv2.IMREAD_GRAYSCALE)


h, w = src1.shape

print(src1.shape)
src2 = np.zeros((h,w), dtype=np.uint8)

for x in range(int(w/2)):
    for y in range(h):
        src2[y,x] = 255

#cv2.rectangle(src2, (0,0), (320,640), (255,255,255), -1)

#dst = cv2.bitwise_and(src1, src2, dst=None, mask=None)
dst = cv2.bitwise_or(src1, src2, dst=None, mask=None)
#dst = cv2.bitwise_xor(src1, src2, dst=None, mask=None)
#dst = cv2.bitwise_not(src1, dst=None, mask=None) # 원본 이미지에 영향을 줌


cv2.imshow('src1',src1)
cv2.imshow('src2',src2)
cv2.imshow('dst', dst)

cv2.waitKey()
cv2.destroyAllWindows()