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

'''
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
'''

###############################################
#########   grabcut 응용


import numpy as np
import cv2


# 입력 영상 불러오기
src = cv2.imread('./images/ob/messi.jpg')

# 사각형 지정을 통한 초기 분할
mask = np.zeros(src.shape[:2], np.uint8)  # 마스크

# mask 를 지정하는 방법
bgdModel = np.zeros((1, 65), np.float64)  # 배경 모델
fgdModel = np.zeros((1, 65), np.float64)  # 전경 모델

rc = cv2.selectROI(src)

cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
dst = src * mask2[:, :, np.newaxis]

# 초기 분할 결과 출력
cv2.imshow('dst', dst)

# 마우스 이벤트 처리 함수 등록
def on_mouse(event, x, y, flags, param):
    # 왼쪽 버튼이 눌러지면 foreground
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)
        cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
        cv2.imshow('dst', dst)
    # 오른쪽 버튼이 눌러지면 foreground
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)
        cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
        cv2.imshow('dst', dst)
    elif event == cv2.EVENT_MOUSEMOVE:
        # 왼쪽 버튼이 눌러져 있는 경우에도 foreground 검
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(dst, (x, y), 3, (255, 0, 0), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
            cv2.imshow('dst', dst)
        # 오른쪽 버튼이 눌러져 있는 경우에도 foreground 검
        elif flags & cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(dst, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
            cv2.imshow('dst', dst)

cv2.setMouseCallback('dst', on_mouse)

while True:
    key = cv2.waitKey()
    if key == 13:  # ENTER
        # 위에서 지정한 ROI 에서 bgdModel, fgdModel 를 이용해서 영상 분할
        cv2.grabCut(src, mask, rc, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        dst = src * mask2[:, :, np.newaxis]
        cv2.imshow('dst', dst)
    elif key == 27:
        break

cv2.destroyAllWindows()


