# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:16:15 2020

@author: ilrma
"""


import csv
import numpy as np
from PIL import Image, ImageFilter
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Polygon



    
def cal_maxmin(csv_file):
    with open(csv_file) as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    l = l[1:]
    l = [list(a) for a in zip(*l)]

    x_axis = l[0]
    y_axis = l[1]
    x_axis = [int(x) for x in x_axis]
    y_axis = [int(y) for y in y_axis]
    ex_val = [min(x_axis),max(y_axis),max(x_axis),min(y_axis)]#x_max,x_min,y_max,y_min
    ex_val = [abs(val) for val in ex_val]
    return ex_val
    
        

def draw_contours_C(file_name,in_dir_name):
    global img
    try:
        img = cv2.imread(file_name)
    except:
        return None

    # 2値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel)
        # 輪郭抽出
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 一番面積が大きい輪郭を抽出
    target_contour = max(contours, key=lambda x: cv2.contourArea(x))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_axis_off()

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 500:
            continue
    else:
        cnt = cnt.squeeze(axis=1)
        ax.add_patch(Polygon(cnt, color="b", fill=None, lw=2))
        ax.plot(cnt[:, 0], cnt[:, 1], "ro", mew=0, ms=4)
        ax.text(cnt[0][0], cnt[0][1], i, color="orange", size="20")     
    plt.close()
        # 輪郭を構成する点を CSV に保存する。
    buf_np = target_contour.squeeze(axis=1).flatten()
    x_list = []
    y_list = []
    for i, elem in enumerate(buf_np):
        if i%2==0:
            x_list.append(elem)
        else:
            y_list.append(elem*(-1))    
    # pandasのSeries型へ一旦変換  
    x_df = pd.Series(x_list)
    y_df = pd.Series(y_list)
    # pandasのDataFrame型へ結合と共に、列名も加えて変換
    DF = pd.concat((x_df.rename(r'#X'), y_df.rename('Y')), axis=1, sort=False)
    if not os.path.isdir('{}\\img_crop'.format(os.path.split(file_name)[0])):
        os.makedirs('{}\\img_crop'.format(os.path.split(file_name)[0]))
    if in_dir_name == None:
        csv_name = os.path.split(file_name)[0] + '\\img_crop\\' +  os.path.split(file_name)[1]
    else:
        csv_name = file_name.replace('original','img_crop\\contour_csv{}'.format(os.path.splitext(os.path.basename(in_dir_name))[1]))
    csv_name = csv_name.replace('.jpg','.csv')
    DF.to_csv(csv_name, encoding="utf-8", index=False)
    print('save as {}'.format(csv_name))
    return csv_name

def Cropping_Image(in_dir_name):
    gyudon_img = []
    root_dir = os.path.dirname(os.path.dirname(in_dir_name))
    in_file_name = os.path.splitext(os.path.basename(in_dir_name))[0]
    if not os.path.isdir('{}\img_crop\{}'.format(root_dir,in_file_name)):
        os.makedirs('{}\img_crop\{}'.format(root_dir,in_file_name))
    if not os.path.isdir('{}\img_crop\contour_csv\{}'.format(root_dir,in_file_name)):
        os.makedirs('{}\img_crop\contour_csv\{}'.format(root_dir,in_file_name))
    gyudon_img = glob.glob('{}\*.jpg'.format(in_dir_name))
    for file_name in gyudon_img:
        img1 = Image.open(file_name)
        img_csv = draw_contours_C(file_name,in_dir_name)
        img_crop = img1.crop((cal_maxmin(img_csv)))
        img_resize = img_crop.resize((400,400)) 
        img_crop_name = '{}\img_crop\{}'.format(root_dir,in_file_name) + '\\' + os.path.splitext(os.path.basename(file_name))[0] + '_crop' + '.jpg'
        print('cropping image : {}'.format(img_crop_name))
        img_resize.save(img_crop_name,quality=95)
    print('end crop')