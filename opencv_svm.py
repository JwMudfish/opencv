### Support Vector Machines

# digits.png 파일 학습 방법
# • 하나의 숫자는 20x20 크기로 되어있음
# • 20x20 크기의 숫자가 총 5000개 있음
# • 5x5 셀을 2x2 로 묶어 block 을 만듦(즉, 10x10)
# • 하나의 cell 당 9개의 bin 으로 나눔(HOG Descriptor)
# • stride 는 (5,5) 로 하여 가로, 세로 각각 3칸씩 이동 가능
# • 하나의 숫자랑 3x3x4x9 = 324 차원 벡터로 변환
# • 총 5000개의 숫자를 1x324 형태로 변환하여 5000x324 형태 이미지 생성

'''
cv2.ml.SVM_create() -> retval
'''

################################
'''
svm = cv2.ml.SVM_create()
svm.setType(type
'''
# • Type 종류
#     • C_SVC : classification
#     • NU_SVC : classification
#     • ONE_CLASS : 1-class classification
#     • EPS_SVR : Regression
#     • NU_SVR : Regression

###################################
'''
svm = cv2.ml.SVM_create()
svm.ml_SVM.setKernel(kernelType)
'''
# • Kernel 종류 : 함수 선택 및 hyperparameter 선택 필요
#     • cv2.ml.SVM_LINEAR : 선형함수
#     • cv2.ml.SVM_POLY : 다항함수
#     • cv2.ml.SVM_RBF : Radial basis function
#     • cv2.ml.SVM_SIGMOID : sigmoid

###################################
'''
svm = cv2.ml.SVM_create()
svm.trainAuto(trainData, layout, label)
'''
# • trainData : 학습 데이터(np.ndarray)
# • layout : trainData 데이터가 입력된 기준 설정(행/열)
#   • cv2.ml.ROW_SAMPLE
# • responses : label(np.ndarray)

###################################

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

hog = cv2.HOGDescriptor((20,20), (10,10), (5,5), (5,5), 9)

cells = [np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)]
cells = np.array(cells)
cells = cells.reshape(-1, 20, 20)

desc = []
for img in cells:
    desc.append(hog.compute(img))

data = np.array(desc)  # shape = (5000,342,1)
data = data.squeeze().astype(np.float32)  # shape = (5000, 325)

train_labels = np.repeat(np.arange(10), len(data)/10)

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)

### training 하기
#svm.trainAuto(data, cv2.ml.ROW_SAMPLE, train_labels)
#print('최적의 C 값 : ', svm.getC())  # 2.5
#print('최적의 Gamma 값 : ', svm.getGamma())   # 0.50625000000001

# training 결과 사용하기
svm.setC(2.5)
svm.setGamma(0.5062500000000001)
svm.train(data, cv2.ml.ROW_SAMPLE, train_labels)


img = np.zeros((400,400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img',mouse_fn)

while True:
    key = cv2.waitKey()

    if key == 27:
        break

    elif key == ord(' '):
        test_image = cv2.resize(img, (20,20), cv2.INTER_AREA)
        test_desc = hog.compute(test_image).T

        _, res = svm.predict(test_desc)
        print(int(res[0, 0]))


        img.fill(0)
        cv2.imshow('img', img)

cv2.destroyAllWindows()








