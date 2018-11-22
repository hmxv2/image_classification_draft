'''
功能：识别出图片中的文字区域占比多大。以此区分是否是文字为主的图片。

思路：尝试借用现成的文字识别工具tesseract，该工具是运行在window平台的一个exe程序，使用python调用exe，返回文字识别结果，速度较慢，一张图的处理速度大约是50s。

作者：miokinhuang

时间：2018.8.23

备注：对于我们这个任务，识别出文字内容不是必须的，只要能检测出是文字区域即可，不需要知道具体是什么内容。尝试其他途径。该脚本目前没有启用，等以后有必要识别出具体文字时才派得上用场。

'''


from PIL import Image, ImageOps
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt
import random
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

#def PIL_img_show(img):
#    Image.fromarray(img).show()

def contours_filter(contours, hierarchy):#汉字的一个特点是文字内部有空洞，由此形成连通域，但是检测句子所在行的话应该去掉这部分
    first_child = hierarchy[0][0][2]
    next_child = first_child
    all_children=[]
    while(next_child!=-1):
        all_children.append(next_child)
        next_child = hierarchy[0][next_child][0]
    contours_filtered = [contours[x] for x in all_children]
    
#     print(len(all_children), all_children)
    return contours_filtered

def get_text_rate(bin_img, boxs_horizontal):
    
    img_heigth, img_width = bin_img.shape
    small_point_cnt=0
    large_not_text_cnt=0
    small_not_text_cnt=0
    all_text_str=[]
    all_ROI_area_rate=[]
    text_rate_metric=0

    for idx, box_horizontal in enumerate(boxs_horizontal):
        x1,y1=box_horizontal[0]
        x2,y2=box_horizontal[2]
        img_ROI = bin_img[y1:y2, x1:x2]
        ROI_area = (x2-x1)*(y2-y1)
        ROI_area_rate = ROI_area/img_heigth/img_width


        if ROI_area/img_heigth/img_width<0.0005:#是小的噪点，不是中文或字母
            small_point_cnt+=1
        elif ((sum(sum(1-img_ROI/255)))/ROI_area)>0.7:#黑色部分太大，属于大片黑色非文字区域
            large_not_text_cnt+=1
        else:
            text_str = pytesseract.image_to_string(Image.fromarray(img_ROI), lang='chi_sim')
            if text_str=='' or text_str ==' ' or text_str=='\t':
                small_not_text_cnt+=1
            else:
                #print(idx, text_str, ROI_area_rate, ROI_area_rate)
                all_text_str.append(text_str)
                all_ROI_area_rate.append(ROI_area_rate)
    #print(small_point_cnt, large_not_text_cnt, small_not_text_cnt)

    
    ROI_area_weights=np.array(all_ROI_area_rate/sum(all_ROI_area_rate))
    all_test_length=[len(x) for x in all_text_str]
    length_div_area=np.array(all_test_length)/np.array(all_ROI_area_rate)

    text_rate_metric = np.mean((length_div_area-length_div_area.mean())**2*ROI_area_weights)
    return text_rate_metric

#     for i in range(len(all_text_str)):
#         text_rate_metric+=len(all_text_str[i])/all_ROI_area_rate[i]

#     return text_rate_metric/len(all_text_str)

def get_likely_text_area(bin_img):
    
    img_height, img_width = bin_img.shape[:2]
    
    kernel_size = (int(img_height/400), int(img_width/1000))
    kernel = np.ones(kernel_size,np.uint8)
    dilate_img = cv2.dilate(bin_img,kernel,iterations = 1)
#     PIL_img_show(dilate_img)

    kernel_size = (int(img_height/400), int(img_width/80))
    kernel = np.ones(kernel_size,np.uint8)
    erosion_img = cv2.erode(dilate_img,kernel,iterations = 2)
    
    image, contours, hierarchy = cv2.findContours(erosion_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_SIMPLE

#     tmp_img = erosion_img-erosion_img
#     cv2.drawContours(tmp_img,contours,-1,(100,100,100),5)
#     PIL_img_show(tmp_img)

#     print(len(contours))
    contours_filtered = contours_filter(contours, hierarchy)
#    print(len(contours_filtered))
    areas=[]
    rects=[]#矩形信息包括：矩形中心，高度，宽度，旋转角
    boxs=[]#矩形信息包括：矩形四个点的坐标，经过rects信息计算而来，经过了计算把浮点数转化为图像坐标，矩形可以不是水平的。

    for cnt in contours_filtered:
        area = cv2.contourArea(cnt)
        areas.append(area)

        rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        rects.append(rect)

        box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
        box = np.int0(box)
        boxs.append(box)

#     tmp_img = erosion_img-erosion_img
#     cv2.drawContours(tmp_img,boxs,-1,(100,100,100),5)
#     PIL_img_show(tmp_img)
    
    boxs_horizontal=[]#将boxs化为方向水平，但是如果句子是倾斜的则圈出的范围过于宽泛。
    for box in boxs:
        x1=min(box[0][0], box[1][0], box[2][0], box[3][0])
        x2=max(box[0][0], box[1][0], box[2][0], box[3][0])
        y1=min(box[0][1], box[1][1], box[2][1], box[3][1])
        y2=max(box[0][1], box[1][1], box[2][1], box[3][1])
        boxs_horizontal.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))

#     print(len(boxs_horizontal))
    return boxs_horizontal, boxs

def dilate_and_erode(bin_img):
    img_height, img_width = bin_img.shape[:2]
    
    kernel_size = (int(img_height/400), int(img_width/1000))
    kernel = np.ones(kernel_size,np.uint8)
    dilate_img = cv2.dilate(bin_img,kernel,iterations = 1)
#     PIL_img_show(dilate_img)

    kernel_size = (int(img_height/400), int(img_width/80))
    kernel = np.ones(kernel_size,np.uint8)
    erosion_img = cv2.erode(dilate_img,kernel,iterations = 2)
    
    return erosion_img