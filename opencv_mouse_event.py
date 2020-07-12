
############# 마우스 이벤트 ############
# cv2.setMouseCallback(windowName, mouse_fn, param=None)
# windowName : 마우스 이벤트를 실행할 윈도우 이름
# onMouse - 마우스이벤트처리를위한콜백함수이름. 마우스 이벤트 콜백 함수는 다음 형식을 따라야 함

# mouse(event, x, y, flags, param)
# param : 콜백 함수에 전달할 데이터
# event : 마우스 이벤트 type
# x, y : 마우스 이벤트가 발생한 x, y 좌표
# flag : 마우스 이벤트 발생시 상태


import cv2
import numpy as np


######## 영상을 마우스로 드레그한 영역을 grayscale 로 변환하여 새로운 윈도우에 띄우는 함수 ###

mouse_is_pressing = False
start_x, start_y = -1, -1


def mouse_callback(event, x, y, flags, param):

    global mouse_is_pressing, start_x, start_y

    img_result = img_color.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_is_pressing = True
        start_x, start_y = x, y
        cv2.circle(img_result, (x,y), 10, (0,255,0), -1)

        cv2.imshow('img_color', img_result)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_is_pressing:
            cv2.rectangle(img_result, (start_x, start_y), (x,y), (0,255,0), 3)
            cv2.imshow('img_color', img_result)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_is_pressing = False

 
        img_cat = img_color[min(start_y,y) : max(start_y,y), min(start_x,x) : max(start_x,x)]

        
        img_cat = cv2.cvtColor(img_cat, cv2.COLOR_BGR2GRAY)
        img_cat = cv2.cvtColor(img_cat, cv2.COLOR_GRAY2BGR)

        img_result[min(start_y,y) : max(start_y,y), min(start_x,x) : max(start_x,x)] = img_cat
        cv2.imshow('img_color', img_result)
        cv2.imshow('img_cat', img_cat)


#img_color = cv2.imread('./images/apple.jpg')
#cv2.imshow('img_color', img_color)

#cv2.setMouseCallback('img_color', mouse_callback)

######## 마우스가 지나간 길을 따라 선을 그려주는 프로그램을 작성하세요 ###

old_x = old_y = -1

def mouse_fn(event, x, y, flags, param):
    global img, old_x, old_y

    if event == cv2.EVENT_LBUTTONDOWN:
        old_x, old_y = x, y

        print('EVENT_LBUTTONDOWN : %d, %d' % (x,y))

    elif event == cv2.EVENT_LBUTTONUP:
        print('EVENT_LBUTTONUP : %d, %d' % (x,y))
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            #print('EVENT_MOUSEMOVE : %d, %d' % (x,y))
            #cv2.circle(img, (x,y), 5, (0,0,255), -1)
            cv2.line(img, (old_x, old_y), (x,y), (0,0,255), 4, cv2.LINE_AA)
            cv2.imshow('img', img)

            old_x, old_y = x, y






img = np.ones((480, 640, 3), dtype=np.uint8) * 255

cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_fn)





cv2.waitKey()
cv2.destroyAllWindows()