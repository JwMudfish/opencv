#################### Sharping #####################
# 경계가 모호한 영상을 경계를 더 뚜렷하게 하는것

# • Isotropic Filter: 영상 내에서 불연속점의 방향에 독립적인 응답,
# • 영상을 회전시키고 난 후, 필터를 적용한 결과와 먼저 필터를 적용하고 그 후 결과를 회전시키는 것과 동일한 결과를 나타낸다. (Rotation Invariant)

# Unsharp mask Filtering
# • unsharp(경계가 뚜렷하지 않은) 영상을 뚜렸하게 변형
# 1.원래 영상 : f(x)
# 2.smoothing 된 영상 : g(x)
# 3.원래 영상 – smoothing 된 영상 : f(x) - g(x)
# 4.원래 영상 + unsharp mask : f(x) + (f(x) - g(x))


################################################
# Unsharp mask filter 구현
# 1 . cv2.GaussianBlur 를 이용해서 blurring
# 2 . 2 * f(x) – blur(f(x)) 함수를 적용
'''
import cv2
import numpy as np

src = cv2.imread('./images/nan.jpg')
src = cv2.resize(src, dsize=(0,0), fx=0.3, fy=0.3)

src_f = src.astype(np.float32)  # float 형태로 변경, 정수인 경우에는 가우시안 필터가 제대로 적용 안됨
blur = cv2.GaussianBlur(src_f, (0,0), 3)   # (0,0) 커널사이즈 정하지 않음, 표준편차 3
m_blur = cv2.blur(src, (9,9))

dst1 = np.clip(2. * src_f - blur, 0, 255).astype(np.uint8)   # clip 값의 상하한 지정, 앞에 2를 곱하는 이유는 경계를 더 뚜렷하게 만들기 위함
dst2 = np.clip(2. * src_f - m_blur, 0, 255).astype(np.uint8)   # clip 값의 상하한 지정


cv2.imshow('src', src)
cv2.imshow('gaussian', dst1)
cv2.imshow('min', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''

##############################################
#############   unsharp mask 의 정도 조절
# • 원래 영상 + unsharp mask : f(x) + (f(x) - g(x))
# • 뚜렷한 결과를 위해서 unsharp mask 의 weight 를 증가시킬 수 있음
# • f(x) + k(f(x) - g(x)) = (k+1)f(x) – k g(x)
'''
import cv2
import numpy as np

src = cv2.imread('./images/nan.jpg')
src = cv2.resize(src, dsize=(0,0), fx=0.3, fy=0.3)

src_f = src.astype(np.float32)  # float 형태로 변경, 정수인 경우에는 가우시안 필터가 제대로 적용 안됨
blur = cv2.GaussianBlur(src_f, (0,0), 3)   # (0,0) 커널사이즈 정하지 않음, 표준편차 3
m_blur = cv2.blur(src, (9,9))


k = 3.
dst1 = np.clip((k+1) * src_f - k * blur, 0, 255).astype(np.uint8)   
dst2 = np.clip((k+1) * src_f - k * m_blur, 0, 255).astype(np.uint8)  


cv2.imshow('src', src)
cv2.imshow('gaussian', dst1)
cv2.imshow('medium', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
'''

##################################################
# 컬러영상의 Unsharp mask filter 적용 방법
# 1 . BRG 을 YCrCb 로 바꾼다
# 2 . 빛에 해당하는 Y 부분만 Unsharp mask filter 를 적용한다.


import cv2
import numpy as np

src = cv2.imread('./images/nan.jpg')
src = cv2.resize(src, dsize=(0,0), fx=0.3, fy=0.3)
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

src_f = src_ycrcb[:,:,0].astype(np.float32)
blr = cv2.GaussianBlur(src_f, (0,0), 9)
blr2 = cv2.blur(src_f, (3,3))

src_ycrcb[:,:,0] = np.clip(2 * src_f - blr, 0, 255).astype(np.uint8)
dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2BGR)



'''
src_g = cv2.split(src)
#src_g[0] = src_g.astype(np.float32)

blur = cv2.GaussianBlur(src_g[0], (0,0), 3)

src_g[0] = np.clip(2. * src_g[0] - blur, 0, 255).astype(np.uint8)

dst = cv2.merge(src_g)
dst = cv2.cvtColor(src_ycrcb, cv2.COLOR_YCrCb2BGR)
'''
# k = 3.
# dst1 = np.clip((k+1) * src_f - k * blur, 0, 255).astype(np.uint8)   
# dst2 = np.clip((k+1) * src_f - k * m_blur, 0, 255).astype(np.uint8)  


cv2.imshow('src', src)
cv2.imshow('gaussian', dst)
# cv2.imshow('medium', dst2)

cv2.waitKey()
cv2.destroyAllWindows()












