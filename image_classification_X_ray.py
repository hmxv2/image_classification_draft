'''
功能：判断图片中是否有超声波图或X光图

思路：超声波图和X光图有一个特点是，大面积的黑色聚集区域。根据这个特点，从图片中查找出所有连通域，排除汉字内部内部围成连通域之后，针对连通域进行排序，找出前K个明显大的连通域，判断其是否超过某个界定阈值，是则可能是黑色超声波图。另外的，由于敏感信息遮挡时使用的也是黑色大方块，故可能和黑色超声波图混淆，需要进一步区分出敏感信息遮挡，结合遮挡区域的特点，即长宽分别平行于图片的长宽，而且区域内部颜色均匀，不会出现白色噪点，故可以根据区域中黑像素的纯度，排除敏感信息遮挡区域，找出普通超声波图和X光图。

作者：miokinhuang

时间：2018.8.20

备注：无

'''

from PIL import Image, ImageOps
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt

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

def get_large_contours(bin_img):#must be dilated
    img_height, img_width = bin_img.shape[:2]
    
    kernel_size = (int(img_height/500), int(img_width/150))#经验值。水平因为文字距离近，所以腐蚀多一点，垂直因为文字离得远，腐蚀少一点也无大碍
    kernel = np.ones(kernel_size,np.uint8)
    dilate_img = cv2.dilate(bin_img,kernel,iterations = 1)
    
    # cv2.RETR_EXTERNAL表示只检测外轮廓
    # cv2.RETR_LIST检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE建立一个等级树结构的轮廓。
    image, contours, hierarchy = cv2.findContours(dilate_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#CHAIN_APPROX_SIMPLE

    contours_filtered = contours_filter(contours, hierarchy)
    if contours_filtered==[]:
        return [],[]
    
    
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
        
        rect_area = max(abs(x2-x1), abs(y2-y1))*min(abs(x2-x1),abs(y2-y1))
        if rect_area/img_height/img_width<0.0005:#太小的区域，只能是噪点和零星的文字，忽略掉
            continue
        boxs_horizontal.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))
    
    if boxs_horizontal==[]:
        return [],[]
#     print(len(boxs_horizontal))

    areas=[]
    for cnt in boxs_horizontal:
        areas.append( cv2.contourArea(cnt))
    
    zipped = zip(boxs_horizontal, areas)
    zipped_sorted = sorted(zipped, key=lambda x:x[1], reverse=True)
    boxs_hor_sorted, areas_sorted = zip(*zipped_sorted)
    return list(areas_sorted), list(boxs_hor_sorted)
    
#     areas_sorted = areas.copy()
#     areas_sorted.sort()
#     areas_sorted = [x/(img_height*img_width)*100 for x in areas_sorted]

#     return areas_sorted, boxs_horizontal

def is_X_ray(bin_img, areas_sorted, boxs_hor_sorted, topK_chosen_rate=0.8, topK_average_threshold=3.5):#多个大的连通区域
    if areas_sorted==[]:
        return 'no_black_contour'
        
    topK_num_sum = sum(areas_sorted[i] if x>areas_sorted[0]*0.8 else 0 for (i, x) in enumerate(areas_sorted))#yes 0.8 is a magic number
    topK_idxs = [i for (i, x) in enumerate(areas_sorted)  if x>areas_sorted[0]*0.8 ]
    topK_num_cnt = len(topK_idxs)
    #print('topK_num_cnt:', topK_num_cnt)
    
    if topK_num_sum/topK_num_cnt>topK_average_threshold:#超过了阈值（经验值3.5），说明存在很大的连通区域
        for topK_idx in topK_idxs:
            topK_box = boxs_hor_sorted[topK_idx]
            black_area_rate = black_area_in_box(bin_img, topK_box)
            #print(black_area_rate)#黑色连通域的黑色像素比例可能高达0.98
            if black_area_rate>0.999:#敏感信息遮挡的话是纯黑背景，因为是后期加上的，所以矩形方框是平行于图像，故黑色面积和连通区域比例是100%
                return 'sensitive_info'
            elif is_black_stripe(bin_img, topK_box, is_stripe_threshold=7)==1:
                return 'has_black_stripe'
            else:
                return 'has_black_contour'
    else:
        return 'no_black_contour'

def black_area_in_box(bin_img, box_hor):#一个方框中的黑色像素的数量
    x1,y1=box_hor[0]
    x2,y2=box_hor[2]
    
    blk_img = bin_img[y1:y2, x1:x2]
    img_height, img_width = blk_img.shape[:2]
    return sum(sum(1-blk_img/255))/img_height/img_width

def is_black_stripe(bin_img, box_hor, is_stripe_threshold=7):#是否是分割线，特征是黑色长条状，设定长宽比例大于7的是分割线
    x1,y1=box_hor[0]
    x2,y2=box_hor[2]
    
    rect_length=max(abs(y2-y1),abs(x2-x1))
    rect_width=min(abs(y2-y1),abs(x2-x1))
    if rect_length/rect_width>is_stripe_threshold:#长宽差距很大，则是黑色分割条
        #print('长宽差距很大')
        return 1
    else:
        return 0