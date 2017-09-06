#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from math import log
import operator


def createDate():
    dataset = [
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']]
    labels = ['no surfing','flippers']
    return dataset,labels


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classList.keys():
            classList[vote] = 0;
        classList[vote] +=1
    sorteClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorteClassCount[0][0]


def calcShannonEnt(dataset):#计算信息熵
    cnt = len(dataset)
    labelCounts = {}
    Ent = 0.0
    for var in dataset:
        labels = var[-1]
        if labels not in labelCounts.keys():
            labelCounts[labels] = 0
        labelCounts[labels] += 1
    for key in labelCounts:
        valnum = float(labelCounts[key])/cnt
        Ent -= (valnum*log(valnum,2))
    return Ent


def splitDataset(dataset,axis,value):
    resData = []
    for var in dataset:
        if var[axis]==value:
            tmplist = var[:axis]
            tmplist.extend(var[axis+1:])
            resData.append(tmplist)
    return resData


def chooseBestdiv(dataset):
    cnt = len(dataset[0])-1
    baseEntropy = calcShannonEnt(dataset)  # 计算为划分的信息熵
    bestEntropy = 0.0;bestchoice = -1
    for i in range(cnt):
        tmplist = [example[i] for example in dataset]
        uniquelist = set(tmplist)
        newEntropy = 0.0
        for value in uniquelist:
            subDataset = splitDataset(dataset,i,value)
            pro = len(subDataset)/float(len(dataset))
            newEntropy += pro*calcShannonEnt(subDataset)
        increaseEntropy = baseEntropy-newEntropy
        if increaseEntropy>bestEntropy:
            bestEntropy = increaseEntropy
            bestchoice = i
    return bestchoice


def createTree(dataset,labels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataset[0])==1:
        return majorityCnt(classList)
    bestFeature = chooseBestdiv(dataset) #选取最好的特征
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    featValues = [example[bestFeature] for example in dataset]
    uniqueVal = set(featValues)
    for value in uniqueVal:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataset(dataset,bestFeature,value),subLabels)
    return myTree


if __name__== '__main__':
    dataset,labels = createDate()
    # print dataset
    # print labels
    myTree = createTree(dataset,labels)
    print myTree

