'''
功能：总体调用多个脚本接口，判断一个图片的类型。

思路：设计分类规则的类CLASSIFY_RULE，初始化一些重要阈值，对外提供接口classify，接受一张图片并且根据制定的规则，判断并输出图片的分类。规则可以进一步优化，阈值建议不修改。

作者：miokinhuang

时间：2018.8.23-24

备注：用法python image_classification.py [input_image_folder] [output_classification_folder] eg: python image_classification.py ./output/ ./results/
其中results文件夹需要事先建立。

'''

from PIL import Image, ImageOps
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import shutil

from image_classification_X_ray import get_large_contours
from image_classification_X_ray import is_X_ray

from image_classification_StraightLine import get_straight_line_cnt

from image_classification_column_hist import get_column_hist_and_fitting

from image_classification_text_using_SWT import text_area_div_likely_text_area

results_set=['text', 'form', 'X-ray', 'others']

img_path = sys.argv[1]
classify_results_path = sys.argv[2]
#判断是否存在
isExists=os.path.exists(img_path)
if not isExists:
    print('path %s is not exist.'%img_path)
    sys.exit()
isExists=os.path.exists(classify_results_path)
if not isExists:
    print('path %s is not exist.'%classify_results_path)
    sys.exit()

def make_folder_for_all_classify_result(results_set):
    for result in results_set:
        isExists=os.path.exists(classify_results_path+result)
        if not isExists:
            os.makedirs(classify_results_path+result)
            print('folder %s has been made.'%result)
        else:
            print('folder %s is exist.'%result)

class CLASSIFY_RULE:
    def __init__(self, LINE_CNT__THRESHOLD, TEXT_RATE_THRESHOLD):#经验值分别为9和1.0
        #['no_black_contour', 'has_black_contour', 'has_black_stripe', 'sensitive_info', 'unk']
        self.LINE_CNT__THRESHOLD = LINE_CNT__THRESHOLD
        self.TEXT_RATE_THRESHOLD = TEXT_RATE_THRESHOLD
        
    def classify(self, img_id, black_area_classify_result, straight_lines_cnt, text_rate):
        if text_rate>=self.TEXT_RATE_THRESHOLD:
            classify_result = 'text'
        elif black_area_classify_result == 'has_black_contour':
            classify_result = 'X-ray'
        elif straight_lines_cnt>=self.LINE_CNT__THRESHOLD or black_area_classify_result == 'has_black_stripe':
            classify_result = 'form'
        else:
            classify_result = 'others'
            
        return classify_result
    
    

start_time=time.time()#计时开始
print('running...')

classify_rule = CLASSIFY_RULE(LINE_CNT__THRESHOLD=8, TEXT_RATE_THRESHOLD=1.0)


#支持读取的图片不是按照img_10000xx.png格式命名，支持命名的数字非连续，支持png之外的其他图片格式。支持文件夹里面有其他格式的文件，例如txt。
make_folder_for_all_classify_result(results_set)
likely_img_file = os.listdir(img_path)
for img_file in likely_img_file:
    if img_file[-3:] not in ['bmp','png','jpg', 'jpeg', 'BMP', 'PNG', 'JPG', 'JPEG']:#非该后缀名列表中的文件不处理，图片格式不够可以加
        continue
    else:
        img = cv2.imread(img_path + img_file)
    
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = gray_img.shape[:2]

        bin_threshold, bin_img = cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #计算连通域，并且进行滤波，每个汉字中笔画包围的小区域去除，保留外层连通域，并进行从大到小的排列
        areas_sorted, boxs_hor_sorted = get_large_contours(bin_img)
        black_area_classify_result = is_X_ray(bin_img, areas_sorted, boxs_hor_sorted)

        #下面注释的代码可以画出当前找到的连通域，很花时间，非调试不使用
#         tmp_img = gray_img - gray_img
#         cv2.drawContours(tmp_img,boxs_hor_sorted,-1,(100,100,100),5)
#         PIL_img_show(tmp_img)
    
        #计算图像中的直线数量，有阈值设定，在image_classification_StraightLine.py文件中。已经调节好，不轻易更改
        straight_lines_cnt = get_straight_line_cnt(bin_img)

        #计算文字的占有率，即文字区域面积除以总的连通域面积，经过尝试是文字居多的图片该值大于1
        text_rate = text_area_div_likely_text_area(bin_img, gray_img)

        black_area_classify_set = ['no_black_contour', 'has_black_contour', 'has_black_stripe', 'sensitive_info', 'unk']
        #print(img_id, black_area_classify_result, straight_lines_cnt, text_rate)

        #按规则分类，规则在 CLASSIFY_RULE 类中定义
        classify_result = classify_rule.classify(img_file, black_area_classify_result, straight_lines_cnt, text_rate)
        print(img_file, classify_result)

        #经过测试，此处复制消耗时间大约是0.1秒，不影响性能
        shutil.copyfile(img_path+img_file, classify_results_path+classify_result+'/'+img_file)      #复制文件

        #下面的指标需要进一步设计，目前的鲁棒性不算很高，一般般，无法完全区分出表格和某些文字，to do
        #curve_fitting_res, has_several_wave, _ = get_column_hist_and_fitting(bin_img)
        #print(img_id, black_area_classify_result, straight_lines_cnt, curve_fitting_res, has_several_wave, text_rate_metric)

#计算总体耗时
#实际实践中，具体消耗是大约一张1M的图片处理耗时4.5秒。
print('running time: %5.2f minutes.'%((time.time()-start_time)/60))