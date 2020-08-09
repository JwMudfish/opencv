############  K-Nearest Neighbor(K-최근접 이웃)
# • Classification Algorithm
# • 근접한 K 개의 data 의 class 별 비율을 기준으로 새로운 data 의 class 를 결정하는 Algorithm

# digits.png 파일 학습 방법
# • 하나의 숫자는 20x20 크기로 되어있음
# • 20x20 크기의 숫자가 총 5000개 있음
# • 20x20 영상으로 나누어서 1x400 형태로 변환
# • 총 5000개의 숫자를 1x400 형태로 변환하여 5000x400 형태 이미지 생성

'''
cv.ml_KNearest.findNearest(samples, k, results=None, neighborResponses=None, dist=None , flags=None)
-> retval, results, neighborResponses, dist
'''
# • sample : 입력 영상(data 별로 행 벡터로 저장됨) , np.ndarray, np.float32
# • k : 최근접 이웃 갯수
# • results : 각각의 입력 샘플에 대한 예측 결과
# • np.ndarray. shape=(N, 1), np.float32
# • neighborResponses : 예측에 사용된 k개의 최근접 이웃 클래스 정보 행렬.
# • shape=(N, k), np.float32.
# • dist: 입력벡터와 예측에 사용된 k개의 최근접 이웃과의 거리를 저장한 행렬
# • np.ndarray. shape=(N, k), np.float32
# • retval : 입력벡터가 하나인 경우 결과

'''
### knn 예제 ###
import numpy as np
import matplotlib.pyplot as plt
import cv2

np.random.seed(0)

### 데이터 만들기
data = np.random.randint(0, 100, (25,2)).astype(np.float32)
#print(data)

response = np.random.randint(0,2,(25,1)).astype(np.float32)
#print(response)
#print(response.ravel())  # ravel -> 가로 array로 합치기

#for i in range(25):
#    print(response[i], data[i])

red = data[response.ravel() == 0]
blue = data[response.ravel() == 1]

plt.scatter(red[:,0], red[:,1], 80, 'r', '^')
plt.scatter(blue[:,0], blue[:,1], 80, 'b', 's')
#plt.show()

knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, response)

newdata = np.random.randint(0, 100, (1,2)).astype(np.float32)
plt.scatter(newdata[:,0], newdata[:,1], 80, 'g', 'o')


ret, results, neighbors, distance = knn.findNearest(newdata, 5)

print('result :', results)
print('neighbors :', neighbors)
print('distance :', distance)

plt.show()
'''



##  digits.png 학습
import numpy as np
import cv2

x0, y0 = -1, -1

def mouse_fn(event, x, y, flags, param):
    global x0, y0

    if event == cv2.EVENT_LBUTTONDOWN:
        x0, y0 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        x0, y0 = -1, -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (x0,y0), (x,y), (255,255,255), 40, cv2.LINE_AA)
            x0, y0 = x, y
            cv2.imshow('img', img)



digits = cv2.imread('./images/ml/digits.png', cv2.IMREAD_GRAYSCALE)

h, w = digits.shape[:2]
#print(h,w)

# cells의 shape = (50, 100, 20, 20)
cells = [np.hsplit(row, w/20) for row in np.vsplit(digits, h//20)]

cells = np.array(cells)

train_images = cells.reshape(-1, 400).astype(np.float32)
train_labels = np.repeat(np.arange(10), len(train_images) / 10)

knn = cv2.ml.KNearest_create()
knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)

img = np.zeros((400,400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img',mouse_fn)

while True:
    key = cv2.waitKey()

    if key == 27:
        break

    elif key == ord(' '):
        test_image = cv2.resize(img, (20,20), cv2.INTER_AREA)
        test_image = test_image.reshape(-1, 400).astype(np.float32)

        ret, _, _, _ = knn.findNearest(test_image, 5)
        print(int(ret))

        img.fill(0)
        cv2.imshow('img', img)

