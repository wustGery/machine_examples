#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import operator

def classify0(inX,dataset,labels,k):# inX为输入向量,dataset为已经存在的数据集,labels 为标签,k为k近邻的常数
    datasize = dataset.shape[0]
    diffMat = np.tile(inX, (datasize,1)) - dataset #复制和dataset一样规模的矩阵,然后得到距离之差
    sqDiffMat = diffMat**2 #对距离进行平方
    sqDistances  = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  #求欧式距离
    sortDistances = distances.argsort() #从小到大对距离进行排序，并且返回索引
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortDistances[i]]  #得到距离前K的分类
        classCount[voteIlabel] = classCount.get(voteIlabel,0) +1 #得到的分类+1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def deal_data(path):
    print path
    data = np.loadtxt(path, dtype=np.float, delimiter=None)
    n,m = data.shape
    train_size = np.int(n*0.989)
    train_x = data[:train_size,:3]
    train_y = data[:train_size,-1]
    test_x  = data[train_size:,:3]
    test_y  = data[train_size:,-1]
    # print train_x.shape, train_y.shape, test_x.shape, test_y.shape
    return data, train_x, train_y, test_x, test_y

def autoNum(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDateset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDateset = dataset - np.tile(minVals,(m,1))
    normDateset = normDateset/np.tile(ranges,(m,1))
    return normDateset, ranges, minVals


if __name__ =="__main__":
    path = "/home/wust_gery/PycharmProjects/test/datingTestSet2.txt"
    dataset, train_x,train_y,test_x,test_y = deal_data(path)
    # plt.figure()
    # idx_1 = np.where(train_y==1)
    # print idx_1
    # plt.scatter(train_x[idx_1,1], train_x[idx_1, 2], marker='x', color ='m', label='1', s = 30)
    # idx_2 = np.where(train_y == 2)
    # plt.scatter(train_x[idx_2,1], train_x[idx_2, 2], marker='o', color ='r', label='2', s=30)
    # idx_3 = np.where(train_y == 3)
    # plt.scatter(train_x[idx_3,1], train_x[idx_3, 2], marker='+', color ='g', label='3', s=40)
    # plt.show()

    # plt.figure()
    # idx_1 = np.where(train_y == 1)
    # print idx_1
    # plt.scatter(train_x[idx_1, 0], train_x[idx_1, 1], marker='x', color='m', label='1', s=30)
    # idx_2 = np.where(train_y == 2)
    # plt.scatter(train_x[idx_2, 0], train_x[idx_2, 1], marker='o', color='r', label='2', s=30)
    # idx_3 = np.where(train_y == 3)
    # plt.scatter(train_x[idx_3, 0], train_x[idx_3, 1], marker='+', color='g', label='3', s=40)
    # plt.show()

    # plt.figure()
    # idx_1 = np.where(train_y == 1)
    # print idx_1
    # plt.scatter(train_x[idx_1, 0], train_x[idx_1, 2], marker='x', color='m', label='1', s=30)
    # idx_2 = np.where(train_y == 2)
    # plt.scatter(train_x[idx_2, 0], train_x[idx_2, 2], marker='o', color='r', label='2', s=30)
    # idx_3 = np.where(train_y == 3)
    # plt.scatter(train_x[idx_3, 0], train_x[idx_3, 2], marker='+', color='g', label='3', s=40)
    # plt.show()


    #进行数据归一化  公式是newValue=(oldValue-min)/(max-min)
    normDateset, ranges, minVals = autoNum(train_x)
    # print normDateset.shape[0]

    train_size = train_x.shape[0]#900

    cnt =0
    err = 0.0

    for i in range(train_size):
        class_label = classify0(normDateset[i,:],normDateset[0:train_size,:],train_y,3)
        print "the class_label is: %d, the real_label is： %d"  % (class_label,train_y[cnt])
        if (class_label != train_y[cnt]):
            err +=1.0
        cnt+=1
    print "the total error rate is %f"  %(err/train_size)









