# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:22:27 2020

@author: ilrma
"""
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
import datetime
now = datetime.datetime.now()
now = now.strftime("%y%m%d")

##### 輪郭抽出の手法A
##### 色調に明確に差があり、それが輪郭になり得る場合
##### HSVに変換後に2値化して判別
def draw_contours_A(file_name, img):
    height, width = img.shape[:2]

    # 色調変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imwrite(now + '_1_' + file_name + '_hsv_A.jpg', hsv)

    # ガウス変換
    gauss = cv2.GaussianBlur(hsv,(9, 9),3)    
    #cv2.imwrite(now + '_2_' + file_name + '_gauss_A.jpg', gauss)
    
    # 色調分割
    img_H, img_S, img_V = cv2.split(gauss)
    #cv2.imwrite(now + '_3_' + file_name + '_H_of_HSV_A.jpg', img_H)
    _thre, img_mask = cv2.threshold(img_H, 140, 255, cv2.THRESH_BINARY)
    #cv2.imwrite(now + '_4_' + file_name + '_mask_A.jpg', img_mask)

    # 輪郭抽出
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    x_list = []
    y_list = []
    for i in range(0, len(contours)):
        if len(contours[i]) > 0:
            if cv2.contourArea(contours[i]) < 500:
                continue
            cv2.polylines(img, contours[i], True, (255, 255, 255), 5)
            buf_np = contours[i].flatten() # numpyの多重配列になっているため、一旦展開する。
            #print(buf_np)
            for i, elem in enumerate(buf_np):
                if i%2==0:
                    x_list.append(elem)
                else:
                    y_list.append(elem*(-1))
    
    #cv2.imwrite(now + '_5_' + file_name + '_boundingbox_A.jpg', img)
    # pandasのSeries型へ一旦変換
    x_df = pd.Series(x_list)
    y_df = pd.Series(y_list)
    # pandasのDataFrame型へ結合と共に、列名も加えて変換
    DF = pd.concat((x_df.rename(r'#X'), y_df.rename('Y')), axis=1, sort=False)
    #print(DF)
    DF.to_csv(now + '_' + file_name + "_target_contour_A.csv", encoding="utf-8", index=False)

##### 輪郭抽出の手法B
##### 色調に差があり、それが輪郭になり得る場合
##### 2値化後に閾値で判別
def draw_contours_B(file_name, img):
    # 2値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(now + '_1_' + file_name + '_gray_B.jpg', gray)
    
    ret,th1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    #cv2.imwrite(now + '_2_' + file_name + '_th1_B.jpg', gray)
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            epsilon = 0.1*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            areas.append(approx)
    #cv2.drawContours(img, areas, -1, (0,255,0), 3)
    cv2.imwrite(now + '_3_' + file_name + '_boundingbox_B.jpg', img)
    
##### 輪郭抽出の手法C
##### 色調に明確な差がなく、線で輪郭になり得る場合
##### 2値化後に閾値で判別
def draw_contours_C(file_name, img):
    # 2値化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(now + '_1_' + file_name + '_gray_C.jpg', gray)

    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imwrite(now + '_2_' + file_name + '_otsu_C.jpg', binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, kernel)
    #cv2.imwrite(now + '_3_' + file_name + '_dilate_C.jpg', binary)
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
    #plt.savefig(now + '_4_' + file_name + '_boundingbox_C.png')
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
    #print(DF)
    DF.to_csv(now + '_' + file_name + "_target_contour_B.csv", encoding="utf-8", index=False)

##### 輪郭抽出の手法D
##### 色調に明確な差がなく、線で輪郭になり得る場合
##### Canny(エッジ検出アルゴリズム)で判別
def draw_contours_D(file_name, img):
    edges = cv2.Canny(img, 100, 200)
  
    cv2.imwrite(now + '_' + file_name + '_boundingbox_D.jpg', edges)
##### csvファイルから散布図を作成する関数
def plot_scatter(myfile):
    file_name = myfile[:-4]

    df = pd.read_csv(myfile, header=0)
    #print("df ->" + str(df))
    myX = df.iloc[:, 0].values.tolist()
    myY = df.iloc[:, 1].values.tolist()

    ax = plt.figure(num=0, dpi=120).gca() 
    ax.set_title(file_name, fontsize=14)
    ax.set_xlabel('X Axis', fontsize=16)
    ax.set_ylabel('Y Axis', fontsize=16)
    ax.scatter(myX, myY, s=20, color="red") #, label=file_name)
    ax.plot(myX, myY)
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
    #plt.legend(loc='auto', fontsize=15)
    #plt.tick_params(labelsize=15) 
    plt.savefig(now + '_' + file_name + "_contours.png")
    plt.close()

##### メイン関数
def main(my_file):
    file_name = my_file[:-4]
    img = cv2.imread(my_file)

    # 輪郭描写（自作関数呼び出し）
    #draw_contours_A(file_name, img) # HSVに変換後に2値化して判別    
    #draw_contours_B(file_name, img) # 2値化後に閾値で判別
    draw_contours_C(file_name, img) # 2値化後に閾値で判別 大津アルゴリズム
    #draw_contours_D(file_name, img) # Canny(エッジ検出アルゴリズム)で判別
  
if __name__ == '__main__':
    my_jpg_list = glob.glob("*.jpg")
    for my_file in my_jpg_list:
        try: # 例外処理により、何かしらエラーが出ても次のループへ進む
            main(my_file)
        except:
            print('somthing error')

    # 輪郭座標のcsvファイルを読み込みグラフ化
    my_csv_list = glob.glob("*.csv")
    for my_file in my_csv_list:
        try:
            pass
            plot_scatter(my_file)
        except:
            print('somthing error')