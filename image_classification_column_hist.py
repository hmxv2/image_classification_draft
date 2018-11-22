'''
功能：因为一些表格没有直线，所以需要另外想特征。

思路：这类表格的显著特征是，每一行的空缺处在行与行之间是大致对齐的，据此可以想到，如果将黑色像素按列累加，画出每列的黑像素数量的分布图，如图3所示，分布图呈现出明显的波峰和波谷，如果能进一步描述这些波峰波谷的起伏大小，则可以作为表格的一个识别特征。使用正弦函数和一次函数对分布图进行拟合，用拟合误差来判断图片是否为表格。

作者：miokinhuang

时间：2018.8.21

备注：该方法目前鲁棒性不高，暂时不启用，但是这个一个非常好的思路。目前需要找一个更好的形式函数来拟合分布，当前采用的sinx+x的形式(即以下代码中采用的func1函数)不够灵活，拟合能力有限，但是过于灵活的形式函数又很容易造成过度拟合(例如func3)。

'''



from PIL import Image, ImageOps
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq   # 从scipy库的optimize模块引入leastsq函数

import random

def func1(x, p):
    A, T, theta, b, A2, theta2, A3, theta3 = p
    return A*np.sin(2*np.pi/T*x+theta)+b+A3*x

def func2(x, p):
    A, T, theta, b, A2, theta2, A3, theta3 = p
    return A*np.sin(2*np.pi/T*x+theta)+b+ A2*np.sin(2*2*np.pi/T*x+theta2)+ A3*x

def func3(x, p):
    A, T, theta, b, A2, theta2, A3, theta3 = p
    return A*np.sin(2*np.pi/T*x+theta)+b+ A2*np.sin(2*2*np.pi/T*x+theta2)+A3*np.sin(3*2*np.pi/T*x+theta3)

def residuals(p, y, x):
    """
    get residuals between the fitting curve and true data
    data: x,y
    parameters:p
    """
    return y - func1(x, p) #此处选择哪个函数作为拟合的基础函数。经过尝试，选了f1，复杂度低，只能拟合简单的波峰波谷，带有二次项的话容易造成过度拟合

def fit_curve_and_get_residuals(residuals, x, y, fitting_guess):
    [A, T, theta, b]=fitting_guess
    plsq = leastsq(residuals, [A, T, theta, b, A/10, theta, 1/100, theta], args=(y, x))
#     res = (residuals(plsq[0], y, x)**2).mean()
    
#     #return
#     return (res)/((y-plsq[0][3])**2).mean(), plsq, res

    res = np.abs(residuals(plsq[0], y, x)).mean()#res表示拟合误差
    res_norm = res/np.abs(plsq[0][0])
    #res_norm是为了将拟合误差归一化，因为不同的图片柱状图不一样高，为了res能放在一起比较
    #需要一个和柱状图高度无关的指标。本次的指标除以sin函数的高度，从实践来看效果还行，是否存在更鲁棒的设计？
    return res_norm, plsq, res

def get_column_hist_and_fitting(bin_img):
    
    img_height, img_width = bin_img.shape[:2]
    
    kernel_size = (int(img_height/500), int(img_width/100))#水平方向需要膨胀多一点使得水平的文字连接起来
    kernel = np.ones(kernel_size,np.uint8)
    erosion_img = cv2.erode(bin_img,kernel,iterations = 2)
#     PIL_img_show(erosion_img)

    row_sums=[]
    col_sums=[]
    for x in range(img_height):
        row_sums.append(img_width - sum(erosion_img[x, :])/255)#img_width for inversing pixel
    for y in range(img_width):
        col_sums.append(img_height -sum(erosion_img[:, y])/255)#img_height for inversing pixel
    
    #统计结束，下面开始拟合分布图
    x = np.array(range(len(col_sums)))
    y1 = np.array(col_sums)
    
    res_norm, plsq, res = fit_curve_and_get_residuals(residuals, x, y1, [1000, 700, 0, y1.mean()])
    
    [A, T, theta, b, A2, theta2, A3, theta3]=plsq[0]
    return res_norm, 1.5*T<img_width, plsq[0]    #1.5*T表示该拟合曲线的宽度很小，基本是普通二次函数的形式可以拟合的，没有波峰波谷，故排除
