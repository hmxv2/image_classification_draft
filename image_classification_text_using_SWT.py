'''
功能：识别出图片中的文字占比，方便确定是否是文字为主的图片。

思路：SWT算法可以较好的判断某个区域是否是文字区域，核心思想是借助文字的笔画一般是均匀的这个特点。检测出文字区域之后，要衡量文字的占比，还需要继续检测出图片中的其他类型的区域，根据两个区域的比值大小来确定是否是文字为主。其他区域的检测使用检测连通域的方法。

作者：miokinhuang

时间：2018.8.23-24

备注：图片进行文字区域标记出来的效果很好。检测连通域效果也很好。这两者的比值，经过尝试，大于1.0时图片就是文字为主。

'''


from PIL import Image, ImageOps
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt
import random

import pillowfight


def PIL_img_show(img):
    Image.fromarray(img).show()


def filte_contours(contours, hierarchy):#汉字的一个特点是文字内部有空洞，由此形成连通域，但是检测句子所在行的话应该去掉这部分
    #将连通域进行简单筛选
    #输入：连通域contours和连通域之间的关系Hierarchy
    #输出：筛选之后的新连通域
    #思路：参照hierarchy的定义
    first_child = hierarchy[0][0][2]
    next_child = first_child
    all_children=[]
    while(next_child!=-1):
        all_children.append(next_child)
        next_child = hierarchy[0][next_child][0]
    contours_filtered = [contours[x] for x in all_children]
    
#     print(len(all_children), all_children)
    return contours_filtered


def get_likely_text_area(bin_img):
    #将连通域都提取出来，作为备选的文字区域
    #需要先进行腐蚀和膨胀操作，腐蚀是为了去掉小的噪点，并且把每一行之间分开。
    #膨胀是希望尽量恢复上一步被腐蚀前的图片，同时行的方向过度膨胀，使得同一行文字连接在一起
    #输入：二值化的图像
    #输出：筛选之后的连通域
    img_height, img_width = bin_img.shape[:2]
    
    kernel_size = (int(img_height/300), int(img_width/1000))
    kernel = np.ones(kernel_size,np.uint8)
    dilate_img = cv2.dilate(bin_img,kernel,iterations = 1)
    #PIL_img_show(dilate_img)

    kernel_size = (int(img_height/400), int(img_width/80))
    kernel = np.ones(kernel_size,np.uint8)
    erosion_img = cv2.erode(dilate_img,kernel,iterations = 2)

    #PIL_img_show(erosion_img)
    
    
    image, contours, hierarchy = cv2.findContours(erosion_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_SIMPLE

#     tmp_img = erosion_img-erosion_img
#     cv2.drawContours(tmp_img,contours,-1,(100,100,100),5)
#     PIL_img_show(tmp_img)

#     print(len(contours))
    contours_filtered = filte_contours(contours, hierarchy)
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

    tmp_img = erosion_img-erosion_img
    cv2.drawContours(tmp_img,contours_filtered,-1,(100,100,100),5)
    #PIL_img_show(tmp_img)
    
    boxs_horizontal=[]#将boxs化为方向水平，但是如果句子是倾斜的则圈出的范围过于宽泛。
    for box in boxs:
        x1=min(box[0][0], box[1][0], box[2][0], box[3][0])
        x2=max(box[0][0], box[1][0], box[2][0], box[3][0])
        y1=min(box[0][1], box[1][1], box[2][1], box[3][1])
        y2=max(box[0][1], box[1][1], box[2][1], box[3][1])
        boxs_horizontal.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))

#     print(len(boxs_horizontal))
    return areas, boxs_horizontal, boxs, contours_filtered

#输入：灰度图，因为要保护mask变量，需要二值化为非255的二值图。干脆封装起来避免用户输入的二值化图带有255
#输出：文字区域的面积
def get_text_area(gray_img):#使用swt算法挖掘文字区域，经过测试，该算法准确率极高
    
    img_heigth, img_width = gray_img.shape[:2]
    
    bin_threshold, bin_img = cv2.threshold(gray_img,0,100 ,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#此处取100，只要不取255就行，因为mask用的是255
    
    #将文字区域用黑色矩形框画出，非文字区域用白色画出
    original_img_mask = np.uint8(np.ones((img_heigth,img_width))*255)
    
    bin_img_PIL_formate = Image.fromarray(bin_img)
    
    #bin_img_PIL_formate.show()
    
    swt_out_img = pillowfight.swt(bin_img_PIL_formate, output_type=pillowfight.SWT_OUTPUT_ORIGINAL_BOXES)
    
    #swt_out_img.show()
    
    swt_out_img = np.array(swt_out_img)[:, :, 0]#PIL Image 格式的图像是3通道的，取其中之一即可
    #print(swt_out_img.shape)
    mask_not_text_area=np.uint8(swt_out_img!=original_img_mask)#不相等的区域是文字区域
    #print(mask_not_text_area.shape)
    
    #draw
    #Image.fromarray(255*(1-mask_not_text_area)).show()
    
    
    #mask矩阵求和即是文字区域的面积
    swt_area_sum=sum(sum(np.uint16(mask_not_text_area)))#不用uint16的话会导致第一层sum‘溢出’
    
    return swt_area_sum

def text_area_div_likely_text_area(bin_img, gray_img):
    #需要设计一个指标，能衡量图像中文字的占有比例，以此区分图像为主还是图片表格为主。
    #经过尝试，将SWT方法检测出来的文字区域的面积，除以总的连通域面积，得到的数值视为文字占有率，占有率高的话则是文字为主
    #输入：二值化图，灰度图
    #输出：文字占有率指标
    areas, boxs_horizontal, boxs, contours_filtered = get_likely_text_area(bin_img)
    if areas==[]:
        return 1000000    #分母为0，返回一个大的数字
    else:
        areas_sum = sum(areas)
    
    text_area_sum = get_text_area(gray_img)
    
    return text_area_sum/areas_sum
    