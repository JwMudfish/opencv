# Grabcut
# • 영상내의 픽셀을 그래프로 인식하여 전경(foreground) 와 배경(background) 를 구분하는 알고리즘

'''
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None) -> mask, bgdModel, fgdModel
'''
# • img : 입력 영상
# • mask : 입출력 마스트
#   • cv2.GC_BGD(0),cv2.GC_FGD(1),cv2.GC_PR_BGD(2), cv2.GC_PR_FGD(3)
# • rect :ROI영역
#   • cv2.GC_INIT_WITH_RECT 모드에서만 사용됨
# • bgdModel : 임시배경모델행렬. 같은 영상처리시에는 변경금지.
# • fgdModel : 임시전경모델행렬. 같은 영상처리시에는 변경금지.
# • iterCount : 반복 횟수
# • mode : 보통 cv2.GC_INIT_WITH_RECT모드로 초기화하고, cv2.GC_INIT_WITH_MASK 모드로 업데이트함.

import cv2
import numpy as np

#src = cv2.imread('./images/ob/grabcut_ex.jpg')
src = cv2.imread('./images/man.jpg')

rc = cv2.selectROI(src)
mask = np.zeros(src.shape[:2], np.uint8)

cv2.grabCut(src, mask, rc, None, None, 3, cv2.GC_INIT_WITH_RECT)


mask2 = np.where((mask == 0) | (mask ==2), 0 , 1).astype('uint8')

dst = src * mask2[:,:,np.newaxis]

# 0~3 => 255
mask = mask * 80

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)
cv2.waitKey()
cv2.destroyAllWindows()



