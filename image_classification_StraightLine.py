'''
功能：识别图片中的直线，据此判断图片是否含有表格

思路：目前识别图像直线的经典做法是霍夫变换，将像素转化为参数空间，处于同一直线上的像素具有相同的参数，同时距离较近的直线在参数空间上也具有相近的距离，因此该方法鲁棒性极高。使用霍夫变换，结合本分类任务设定阈值（包括检测直线的最小长度=宽度/2，直线之间的间隔=10 pixel，直线倾角的步进=2度），检测出直线之后，根据直线计数，设定阈值(经过尝试设定为8)判断图片是否含有表格。

作者：miokinhuang

时间：2018.8.21

备注：无

'''



from PIL import Image, ImageOps
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt

def PIL_img_show(img):
    Image.fromarray(img).show()

def get_straight_line_cnt(bin_img):
    img_height, img_width = bin_img.shape[:2]
    
    #借用canny算子，简单提取出边缘
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)
#     PIL_img_show(edges)

#minLineLength是直线的最小长度
#maxLineGap是直线和直线之间如果距离够近的话则视为同一直线
    minLineLength = min(img_height, img_width)/2
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    #lines变量可能是空
    try:
        lines_cnt = len(lines)
    except:
        lines_cnt = 0
    
    #draw image with straight line(s) if lines is not None
    if 0:    #drawing control: 0 means not to draw.
        if lines_cnt!=0:
            tmp_img = bin_img - bin_img
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(tmp_img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            PIL_img_show(tmp_img)

    return lines_cnt