import cv2

img1 = cv2.imread('./images/apple.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/apple.jpg', cv2.IMREAD_COLOR)

print(type(img1)), print(type(img2))

h, w = img2.shape[:2]

'''
if len(img1.shape) == 2:
    print('grayscale 영상입니다.')

elif len(img1.shape) == 3:
    print('컬러 영상입니다.')

print(h,w)
'''

# 특정좌표 컬러 변경
x = 10
y = 20

for x in range(100):
    for y in range(200):
        img2[y,x] = (255,0,0)


img1[y,x] = 0
#img2[y,x] = (255,0,0)


cv2.imshow('img1', img1)
cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()