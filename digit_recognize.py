#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib as mpl


def save_image(im, i):
    im *= 15.9375
    im = 255 - im
    a = im.astype(np.uint8)
    output_path = '.\\HandWritten'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Image.fromarray(a).save(output_path + ('\\%d.png' % i))


def show_accuracy(u,v,label):
    acc = u==v
    w = np.mean(acc)
    print label+'正确率为:%.2f%%' % (w*100)


if __name__ =="__main__":
    print 'Load Training File Start...'
    data = np.loadtxt('14.optdigits.tra', dtype=np.float, delimiter=',')
    x, y =np.split(data,(-1,), axis=1)
    images = x.reshape(-1,8,8)
    y = y.ravel().astype(np.int)

    print 'Load Test Data Start...'

    data = np.loadtxt('14.optdigits.tes', dtype=np.float, delimiter=',')
    x_test, y_test = np.split(data,(-1, ), axis=1)
    images_test = x_test.reshape(-1,8,8)
    y_test = y_test.ravel().astype(np.int)
    print 'Load Data ok...'

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15,9), facecolor='w')
    for index,image in enumerate(images[:16]):#取前面16个图像
        plt.subplot(4,8,index+1)  #4行8列
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'训练图片： %i' % y[index])
    for index, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, index + 17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(image.copy(), index)
        plt.title(u'测试图片: %i' % y_test[index])
    plt.tight_layout()
    plt.show()

    clf = svm.SVC(C=1, kernel='rbf', gamma=0.001)  # C是超参数
    print 'Start Learning'
    clf.fit(x,y)
    print 'Learning is OK...'

    y_hat = clf.predict(x)
    show_accuracy(y,y_hat, '训练集')
    y_hat = clf.predict(x_test)
    show_accuracy(y_test,y_hat,'测试集')
















