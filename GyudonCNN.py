# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:02:57 2020

@author: ilrma
"""


import os
import sys
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class CNN(chainer.Chain):
    def __init__(self, n_out):
        w = I.Normal(scale=0.05) # モデルパラメータの初期化
        super(CNN, self).__init__(
            # Chainerの場合，「層」を記述するのではなく，層と層をつなぐリンク構造を記述する
            # "Linear"は，全結合層（full-connected layer）を表す
            conv1=L.Convolution2D(3, 16, 5, 1, 0), # 1層目の畳み込み層（フィルタ数は16）
            conv2=L.Convolution2D(16, 32, 5, 1, 0), # 2層目の畳み込み層（フィルタ数は32）
            conv3=L.Convolution2D(32, 64, 5, 1, 0), # 3層目の畳み込み層（フィルタ数は64）
            l4=L.Linear(None, n_out, initialW=w), #クラス分類用
        )

    # DNNのフォワード処理を行う（フォワードという言葉の意味も必要）
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2) # 最大値プーリングは2×2，活性化関数はReLU
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), ksize=2, stride=2) 
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2)), ksize=2, stride=2)
        # 9x9,64ch
        return self.l4(h3)
    

def main():
    batchsize = 50
    epoch = 20
    out_dir = 'result'
    data_dir = 'trans_images'
    train = []
    label = 0
    gpu = 1
    print('loading dataset')
    for c in os.listdir(data_dir):
        print('class: {}, class id: {}'.format(c, label))
        d = os.path.join(data_dir, c)        
        imgs = os.listdir(d)
        for i in [f for f in imgs if ('jpg' in f)]:
            train.append([os.path.join(d, i), label])            
        label += 1
    print('')    
    train = chainer.datasets.LabeledImageDataset(train, '.')    
    
    model = L.Classifier(CNN(5)) # CNNにする
    '''if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU''' 
    
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    
    updater = training.StandardUpdater(train_iter, optimizer, device=None)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out_dir)

    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PlotReport('main/loss', 'epoch', file_name='loss.png'))
    trainer.extend(
        extensions.PlotReport('main/accuracy', 'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    # モデルをCPU対応へ
    model.to_cpu()
    # 保存
    modelname = out_dir + "\\FaceEmotion.model"
    print('save the trained model: {}'.format(modelname))
    chainer.serializers.save_npz(modelname, model)

if __name__ == "__main__":
    main()