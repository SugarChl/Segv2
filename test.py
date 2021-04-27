from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import pickle

from osgeo import gdal_array as ga
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cv2 import imread

import numpy as np
import matplotlib.pyplot as plt




def ThreeDtoOneD(img, shape=[2569,1919]):

    def t(gt, a, b, c, t):
        w1 = gt[:, :, 0] == a
        w2 = gt[:, :, 1] == b
        w3 = gt[:, :, 2] == c
        x = w1*w2*w3
        f = np.where(x>0, t, 0)
        return f

    gt = np.zeros(shape)
    gt += t(img, 0, 0, 255, 1)
    gt += t(img, 255, 255, 255, 2)
    gt += t(img, 0, 255, 0, 3)
    gt += t(img, 0, 255, 255, 4)
    gt += t(img, 255, 255, 0, 5)
    return gt



import random
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

def get_error_map(i):
    gt = "D:\Seg\data\Potsdam\gt\\top_potsdam_"+i+"_label_noBoundary.png"
    img = Image.open(gt)
    gt_img = np.array(img)
    print(gt_img.shape)
    pred = str(i) + ".png"
    pred_img = Image.open("C:\\Users\\73876\go\\"+pred)
    pred_img = np.array(pred_img)
    print(pred_img.shape)
    for x in tqdm(range(gt_img.shape[0])):
        for y in range(gt_img.shape[1]):
            if gt_img[x, y, 0] == 0 and gt_img[x, y, 1] == 0 and gt_img[x, y, 2] == 0:
                pred_img[x, y, :] = [0, 0, 0]
                continue
            if pred_img[x, y, 0] != gt_img[x, y, 0] or pred_img[x, y, 1] != gt_img[x, y, 1] or pred_img[x, y, 2] != \
                    gt_img[x, y, 2]:
                pred_img[x, y, :] = [255, 0, 0]
            else:
                pred_img[x, y, :] = [0, 255, 0]
    error_map = Image.fromarray(pred_img)
    error_map.show()
    error_map.save("potsdam_errmap_"+str(i)+".png", 'PNG')


def dsm2hotpic(img):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=plt.cm.hsv)
    c = im.make_image(renderer="colormap", unsampled=True)
    hotpic = Image.fromarray(c[0]).convert("RGB")
    return hotpic



def draw(img):
    # 定义热图的横纵坐标

    data = img
    print(img.shape)
    # 作图阶段
    fig = plt.figure()
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.hsv)

    c = im.make_image(renderer="colormap", unsampled=True)
    print(c[0].shape)
    hotpic = Image.fromarray(c[0]).convert("RGB")
    hotpic.save("D:\\xx.png")
    hotpic.show()
    #return hotpic
    # 增加右侧的颜色刻度条
    #plt.colorbar(im)
    # 增加标题
    #plt.title("This is a title")
    # show
    #plt.show()


import numpy as np


#data = np.load("C:\\Users\\73876\go\Seg\data\ISPRS\dsm\\dsm_09cm_matching_area9.npy")
#draw(data)


#a = np.array([[1, 3, 6, 2],
#              [2, 4, 7, 3],
#              [3, 5, 8, 4],
#              [4, 6, 9, 5]])
#
#print(a)
#
## 垂直翻转
#b = np.zeros_like(a)
#for i in range(a.shape[0]):
#    b[:, a.shape[0]-i-1] += a[:, i]
#print(b)
#
## 水平翻转
#b = np.zeros_like(a)
#for i in range(a.shape[0]):
#    b[a.shape[0]-i-1, :] += a[i, :]
#print(b)

# 两个样本数据密度分布图
#print(img)
##gt_img = ThreeDtoOneD(gt_img)
#
#
#img = np.array(Image.open(path))
#print(img.shape)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#for x in tqdm(range(1250,1350)):
#    for y in range(480,560):
#        if gt_img[x, y] == 1:
#            color = 'red'
#        elif gt_img[x, y] == 2:
#            color = 'blue'
#        elif gt_img[x, y] == 3:
#            color = 'yellow'
#        elif gt_img[x, y] == 4:
#            color = 'green'
#        elif gt_img[x, y] == 5:
#            color = 'black'
#        ax.scatter(x, y, img[x,y], c=color, s=2)
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()

# train_loss, val_loss, val_acc, epoch


#from osgeo import gdal
#import os
#
#files = os.listdir("D:\ISPRS\data\ISPRS_data\dsm")
#for i in files:
#
#    img = gdal.Open(os.path.join("D:\ISPRS\data\ISPRS_data\dsm",i))
#
#    cols=img.RasterXSize#图像长度
#    rows=(img.RasterYSize)#图像宽度
#
#    b = img.GetRasterBand(1).ReadAsArray(0, 0, cols, rows)
#    b = np.array(b)
#    print(i)
#    np.save("D:\Seg\data\ISPRS\dsm\\"+str(i).replace(".tif", "")+".npy", b)


#img = Image.open("D:\ISPRS\data\ISPRS_data\gts_for_participants\\top_mosaic_09cm_area1.tif")

#img = Image.open("D:\ISPRS\data\ISPRS_data\gts_for_participants\\top_mosaic_09cm_area30.tif")

#img_e = Image.open("D:\ISPRS\data\ground_truth\\top_mosaic_09cm_area3_noBoundary.tif")
#
#gt = Image.open("D:\ISPRS\data\ISPRS_data\gts_for_participants\\top_mosaic_09cm_area11.tif")
#
#top_msdfm = Image.open("C:\\Users\\73876\go\Seg\MSDFM_8991\\11.png")
#
#top_msdfmt2 = Image.open("C:\\Users\\73876\go\Seg\MSDFMt2_2019-12-26-12\\11.png")
###img.save("D:\IEEE会议论文模板\\top_30.jpg")
#region = (1000,1500,1550,2400)
#img_e = top_msdfm.crop(region)
#img_e.show()
#img_o = gt.crop(region)
#img_o.show()
#top = top_msdfmt2.crop(region)
#top.show()
#
#
#top.save("C:\\Users\\73876\Desktop\投稿格式\\img\\msdfmt2_11.jpg")
#img_o.save("C:\\Users\\73876\Desktop\投稿格式\\img\\gt_11.jpg")
#img_e.save("C:\\Users\\73876\Desktop\投稿格式\\img\\msdfm_11.jpg")

#gt = "D:\ISPRS\data\ground_truth\\top_mosaic_09cm_area34_noBoundary.tif"
#img = Image.open(gt)
#gt_img = np.array(img)
#print(gt_img.shape)
#pred_img = Image.open("C:\\Users\\73876\go\Seg\MSDFM_8991\\34.png")
#pred_img = np.array(pred_img)
#print(pred_img.shape)
#for x in range(gt_img.shape[0]):
#    for y in range(gt_img.shape[1]):
#        if gt_img[x,y,0] == 0 and gt_img[x,y,1] == 0 and gt_img[x,y,2] == 0:
#            pred_img[x, y, :] = [0, 0, 0]
#            continue
#        if pred_img[x,y,0] != gt_img[x,y,0] or pred_img[x,y,1] != gt_img[x,y,1] or pred_img[x,y,2] != gt_img[x,y,2]:
#            pred_img[x, y, :] = [255, 0, 0]
#        else:
#            pred_img[x, y, :] = [0, 255, 0]
#
#error_map = Image.fromarray(pred_img)
#error_map.show()
#error_map.save("errmap_34.png", 'PNG')
import gdal, os
#from tqdm import tqdm
#
#
#files = os.listdir("D:\Seg\data\Potsdam\\5_Labels_for_participants")
#for file in tqdm(files):
#    print(file)
#    img = gdal.Open(os.path.join("D:\Seg\data\Potsdam\\5_Labels_for_participants", file))
#    cols = img.RasterXSize#图像长度
#    rows = (img.RasterYSize)#图像宽度
#    b0 = np.array(img.GetRasterBand(1).ReadAsArray(0, 0, cols, rows))[:, :, np.newaxis]
#    b1 = np.array(img.GetRasterBand(2).ReadAsArray(0, 0, cols, rows))[:, :, np.newaxis]
#    b2 = np.array(img.GetRasterBand(3).ReadAsArray(0, 0, cols, rows))[:, :, np.newaxis]
#    b = np.concatenate([b0, b1, b2], axis=2)
#    #print(b.shape)
#    img = Image.fromarray(b)
#    #dsm.save("D:\Seg\data\\1_DSM\\"+str(file).replace(".tif", ".png"))
#    img.save("D:\Seg\data\Potsdam\\5_Labels_for_participants\\"+str(file).replace(".tif", ".png").replace("_label", ""))
#
#
#    # top_potsdam_4_12_label.tif

img = gdal.Open("D:\\dsm_potsdam_02_10.tif")
cols = img.RasterXSize  # 图像长度
rows = (img.RasterYSize)  # 图像宽度
b0 = np.array(img.GetRasterBand(1).ReadAsArray(0, 0, cols, rows))[:, :, np.newaxis]

print(b0.shape)  # (6000, 6000, 1)
b0 = b0.squeeze()  # (6000, 6000)
x = dsm2hotpic(b0)
print(x.show())
#x = ["2_11", "2_12", "4_10", "5_11", "6_7", "7_8", "7_10"]
#for i in x:
#    get_error_map(i)
#img = Image.open("C:\\Users\\73876\Desktop\文件\照片\\031702246.jpg")
#out.save(outfile)
#print(img.size)
#img = img.resize((1140//2, 1560//2))
#img.save("C:\\Users\\73876\Desktop\文件\照片\\031702246_.jpg")
