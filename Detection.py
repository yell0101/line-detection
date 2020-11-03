# -*- coding: utf-8 -*-
import cv2
import numpy as np

def grayscale(img):  # grayscale로 변환
    return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

def gaussian_blur(img, kernel_size):  # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) #kernel사이즈는 값이 크게 상관이 없지만 무조건 홀수여야한다.

def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):  # ROI 지정하기

    mask = np.zeros_like(img)  #img와 같은 크기의 빈 이미지 mask 생성

    if len(img.shape) > 2:  # Color 이미지(3채널)라면 :
        color = color3
    else:  # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices의 좌표의 부분을 지정색으로 채움
    cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def mark_img(img, blue_threshold, green_threshold, red_threshold): # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (image[:,:,0] < bgr_threshold[0]) \
                | (image[:,:,1] < bgr_threshold[1]) \
                | (image[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [0,0,0]
    return mark

def max_color(a,b):
    if a>b :
        max=a
    else:
        max=b

    return max

def processRed(img):

    vertices = np.array(
        [[(0, height/3), (width, height/3), (width, 0), (0, 0)]],
        dtype=np.int32)
    ROI_img = region_of_interest(img, vertices)
    imnp = np.array(ROI_img)
    h, w = imnp.shape[:2]
    colors, counts = np.unique(imnp.reshape(-1, 2), axis=0, return_counts=1)
    count = counts[-1]
    proportion = (100 * count) / (h * w)
    if proportion>5:
        print("RED DETECT")
    else:
        print("NO RED")


def processLog(img):


    vertices1 = np.array(
        [[(0, height), (width/2, height), (width/2, 0), (0, 0)]],
        dtype=np.int32)
    vertices2 = np.array(
        [[(width/2, height), (width, height), (width, 0), (width/2, 0)]],
        dtype=np.int32)

    ROI_img1 = region_of_interest(img, vertices1)
    ROI_img2 = region_of_interest(img, vertices2)


    imnp1 = np.array(ROI_img1)
    imnp2 = np.array(ROI_img2)


    h, w = imnp1.shape[:2]
    colors1, counts1 = np.unique(imnp1.reshape(-1, 2), axis=0, return_counts=1)
    colors2, counts2 = np.unique(imnp2.reshape(-1, 2), axis=0, return_counts=1)

    count1 = counts1[-1]
    count2 = counts2[-1]

    proportion1 = (100 * count1) / (h * w)
    proportion2 = (100 * count2) / (h * w)

    print(f"  Color1: {colors1[-1]}, count: {count1}, proportion: {proportion1: .2f}%")
    print(f"  Color2: {colors2[-1]}, count: {count2}, proportion: {proportion2: .2f}%")

    if max_color(proportion1, proportion2) == proportion1:
        print("move to left")
    else:
        print("move to right")




cap = cv2.VideoCapture('test.avi') # 동영상 불러오기

while(cap.isOpened()):
    ret, image = cap.read()
    #image1 = image.copy()
    height, width = image.shape[:2]


    mark = np.copy(image) # roi_img 복사
    mark = mark_img(image,20 ,150, 20) # 흰색 차선 찾기
    mark = gaussian_blur(mark, 9)
    gray_img = grayscale(mark)
    ret, dst = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)

    mark_R = mark_img(image, 20, 20, 170)
    mark_R = gaussian_blur(mark_R, 9)
    gray_img_R = grayscale(mark_R)
    ret1, dst1 = cv2.threshold(gray_img_R, 0, 255, cv2.THRESH_BINARY)

    processRed(dst1)
    processLog(dst)

    cv2.imshow("result3", dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
