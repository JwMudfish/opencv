### K-means Algorithm
# • Clustering Algorithm
# • cluster 갯수를 지정 후 학습 시작
# • 랜덤으로 center 를 지정한 이후에 거리 기반으로 학습
# • cluster 별 center 와의 거리의 합이 최소가 되는 방향으로 학습

# • step 1 : 랜덤으로 중심좌표 지정
# • step 2 : center 와 가장 가까운 cluster 로 지정
# • step 3 : cluster 별 중심 계산하여 새로운 중심으로 지정
# • step 4 : 다시 step2 로 돌아가 새로운 cluseter 의 중심과 다른 데이터와의
#     거리를 계산하여 지정된 학습횟수까지 학습하거나 더이상 움직이지 않을 때까
#     지 학습

'''
cv2.kmeans(data, clusters, criteria, epoch, cv2.KMEANS_RANDOM_CENTERS) -> ret, labels, centers
'''
# • data : 입력 영상(np.ndarrary)
# • clusters : cluster 갯수
# • criteria : 학습 종료 조건
# • (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# • epoch : 학습 횟수
# • cv2.KMEANS_RANDOM_CENTERS : 초기값을 random 으로 선택하여 시작
# • ret : cluster 별 중심과 거리의 제곱 합
# • label : cluster label
# • center : cluster 별 중심 좌표

'''
## kmeans 예제 ######
import numpy as np
import cv2
import matplotlib.pyplot as plt

a = np.random.randint(0, 150, (25,2))
b = np.random.randint(128, 255, (25,2))

data = np.vstack((a,b)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#print(label)

red = data[label.ravel() == 0]
blue = data[label.ravel() == 1]

plt.scatter(red[:,0], red[:,1], c='r')
plt.scatter(blue[:,0], blue[:,1], c='b')

plt.scatter(center[0,0], center[0,1], s = 100, c = 'r', marker = 's')
plt.scatter(center[1,0], center[1,1], s = 100, c = 'b', marker = 's')
plt.show()
'''

import numpy as np
import cv2

img = cv2.imread('./images/ml/taekwonv1.jpg')

K = 16

data = img.reshape((-1,3)).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret, label, center = cv2.kmeans(data, K, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
print(center)

res = center[label.flatten()]
res = res.reshape(img.shape)

merged = np.hstack((img, res))

cv2.imshow('Kmeans Color', merged)
cv2.waitKey()
cv2.destroyAllWindows()