import cv2
import sys

cap = cv2.VideoCapture(0)

w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fps = cap.get(cv2.CAP_PROP_FPS)
fps = 10
print(fps) 

## 코덱 설정
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  # *'MJPG
delay = round(1000 / fps)

# iscolor - 그레이 스케일이면 옵션(마지막 False)을 줘야 저장이 됨
#out = cv2.VideoWriter('200711.avi', fourcc, fps, (w,h))
out = cv2.VideoWriter('200711.avi', fourcc, fps, (w,h), False)


if not cap.isOpened():
    print('camera open failed')
    sys.exit()


#print('frame width : ', cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#print('frame height : ', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()

    if not ret:
        break


    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(frame,50,150)
    edge = ~edge
    
    #out.write(img)
    out.write(edge)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', img)
    cv2.imshow('edge', edge)

    if cv2.waitKey(10) == 27:
        break



cap.release()
out.release()
cv2.destroyAllWindows()
