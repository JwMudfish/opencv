import cv2
import numpy as np

### 키보드 입력 대기
# delay
# m s (밀리세컨즈) 단위 시간( 1초 = 1000)
# delay <= 0 이면 키보드 입력 무한대기
# retval : 눌린 키보드의 ASCII code(ex. 27 : ESC), 눌리지 않으면 -1
# 특정키 입력은 ord() 함수 활용
# 예) while True:
#         if cv.waitKey() == ord('q'):
#             break

##### 키보드 이벤트############
#### 영상을 grayscale 로 변환 후 i 를 누르면 영상의 밝기가 반대로 되는 프로그램

#img = cv2.imread('./images/apple.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('./images/apple.jpg')

#img = ~img
cv2.imshow('image',img)

while True:
    keycode = cv2.waitKey()

    if keycode == ord('i'):
        img = ~img
        cv2.imshow('image', img)
    
    elif keycode == ord('r'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', img)

    elif keycode == 27:
        break

cv2.destroyAllWindows()
