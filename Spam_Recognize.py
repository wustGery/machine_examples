#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import sys


def CreateVocabList(dataset):
    voc = set([])
    for word in dataset:
        voc = voc | set(word)
    return list(voc)


def setOfWords2Vec(vocabList, inputset):
    flag_vec = [0]*len(vocabList)
    for word in inputset:
        if word in vocabList:
            flag_vec[vocabList.index(word)] += 1
        else:
            print "the word: %s is not in vocabList" %word
    return flag_vec


def trainNB0(train_Matrix,train_Classes):
    matrix_len = len(train_Matrix)
    numWords = len(train_Matrix[0])
    p1Num = np.ones(numWords)
    p0Num = np.ones(numWords)
    p1Sum = 2.0
    p0Sum = 2.0
    pAB = sum(train_Classes)/float(matrix_len)
    for i in range(matrix_len):
        if train_Classes[i] == 1:
            p1Num += train_Matrix[i]
            p1Sum += sum(train_Matrix[i])
        else:
            p0Num += train_Matrix[i]
            p0Sum += sum(train_Matrix[i])
    p1Vec = p1Num/p1Sum
    p0Vec = p0Num/p1Sum
    return p1Vec,p0Vec,pAB


def classify_NB0(array_vec,p1Vec,p0Vec,pAB):
    flag_1 = 0.0
    flag_0 = 0.0
    length = len(array_vec)
    for i in range(length):
        flag_1 += array_vec[i]*p1Vec[i]
        flag_0 += array_vec[i]*p0Vec[i]
    print flag_1,flag_0
    if flag_1>flag_0:
        return 1
    else:
        return 0


def textParse(bigString):
    import re
    listOftokens = re.split(r'\W*',bigString)
    return [example.lower() for example in listOftokens if len(example)>2]

def Print(p1Vec,p0Vec,example_voc):
    lenght = len(p1Vec)
    fp = open('out.txt','w')
    for i in range(lenght):
        fp.write("%s:%.8f %.8f\n" %(example_voc[i],p1Vec[i],p0Vec[i]))


def spamTest():
    docList = []; classList = []; fullList = []
    for i in range(1,26):
        wordList = textParse(open('/home/wust_gery/PycharmProjects/test/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullList.append(wordList)
        classList.append(1)#1代表正常的邮件
        wordList = textParse(open('/home/wust_gery/PycharmProjects/test/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullList.append(wordList)
        classList.append(0)#0代表垃圾邮件
    vocabList = CreateVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat = []
    trainingClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p1Vec, p0Vec, pAB = trainNB0(trainingMat, trainingClasses)
    # Print(p1Vec,p0Vec,vocabList)
    errcount = 0
    testingMat = []
    testingLabel = []
    # for key,value in enumerate(testSet):
    #     print key,value
    for key,value in enumerate(testSet):
        test_Vec = setOfWords2Vec(vocabList,docList[value])
        if classify_NB0(test_Vec,p1Vec,p0Vec,pAB)!=classList[value]:
            errcount+=1
    print 'the error rate is :', float(errcount)/len(testSet)


if __name__=='__main__':
    spamTest()
    # example_voc = CreateVocabList(listPosts) #example_voc为所有的单词集合
    # print example_voc
    # train_mat = []
    # for tmp in listPosts:
    #     train_mat.append(setOfWords2Vec(example_voc,tmp))
    # p1Vec,p0Vec,pAB = trainNB0(train_mat,listClasses)
    # lenght = len(p1Vec)
    # fp = open('out.txt','w')
    # for i in range(lenght):
    #     fp.write("%s:%.8f %.8f\n" %(example_voc[i],p1Vec[i],p0Vec[i]))
    # word = [["stupid","garbage","my"],["love","cute","food"],["dog","ate","garbage"]]
    # for key,value in enumerate(word):
    #     word_voc = setOfWords2Vec(example_voc,value)
    #     flag = classify_NB0(word_voc,p1Vec,p0Vec,pAB)
    #     print value," was classified as ",flag





