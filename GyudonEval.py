# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 03:17:44 2020

@author: ilrma
"""


import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image
import chainer
import chainer.functions as F
import chainer.links as L
import CroppingGyudon

cat = ['チー牛','牛丼']

class CNN(chainer.Chain):
    def __init__(self, n_out):
        super(CNN, self).__init__(                        
            conv1=L.Convolution2D(3, 16, 5, 1, 0), # 1層目の畳み込み層（フィルタ数は16）
            conv2=L.Convolution2D(16, 32, 5, 1, 0), # 2層目の畳み込み層（フィルタ数は32）
            conv3=L.Convolution2D(32, 64, 5, 1, 0), # 3層目の畳み込み層（フィルタ数は64）
            l4=L.Linear(None, n_out), #クラス分類用
        )
    
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2)
        return self.l4(h3)
    
def eval_gyudon(image):
    models = 'result//Gyudon.model'
    size = 400
    
    model = L.Classifier(CNN(2)) # モデルの定義
    chainer.serializers.load_npz(models, model) #モデルの読み込み

    img = Image.open(image)
    img_csv = CroppingGyudon.draw_contours_C(image,None)
    img_crop = img.crop((CroppingGyudon.cal_maxmin(img_csv)))
    img_resize = img_crop.resize((size,size))
    img_resize.save('eval.jpg', 'JPEG', quality=100, optimize=True)
    evaldata = np.array(img_resize, dtype=np.float32)
    evaldata = evaldata.transpose(2, 0, 1)
    evaldata = evaldata.reshape(1, 3, size, size) 

    x = chainer.Variable(np.asarray(evaldata))
    y = model.predictor(x) # フォワード
    c = F.softmax(y).data.argmax() 
    res = cat[c]
    print('判定結果は『{}』です。'.format(res))
    return res

