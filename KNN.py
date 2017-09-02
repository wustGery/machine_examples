#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
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


if __name__=="__main__":
    labels = ['A', 'A', 'B', 'B']
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    for i in range(0,10,2):
        flag = classify0([1+i,1+i],group, labels, 3)
        a = np.array([[0+i,0+i]])
        group = np.concatenate((group,a),axis=0)
        labels.append(flag)
        print flag
    print labels
    print group

