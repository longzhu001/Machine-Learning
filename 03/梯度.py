# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 10:34
# @Author  : DONG
# @File    : 梯度.py
# @Software: PyCharm
#coding=utf-8
'''
数据  : 待训练数据A、B为自变量，C为因变量
参数  : x为待预测值的自变量，thta为已经求解出的权重值，yPre为预测结果
      @trainData  : 训练集
      @trainData  : 自变量
      @trainLabel : 最后一列为因变量
      @x   为自变量训练集，
      @y   为自变量对应的因变量训练集；
      @theta  为待求解的权重值，需要事先进行初始化；
      @alpha  是学习率；
      @m  为样本总数；
      @maxIterations  为最大迭代次数
注解  :
    A[ : 2]:表示索引 0至1行；
    A[ :, 2]:表示所有行的第3列。

数据 :  TrainData 训练数据    TestData 测试数据
一般数据会留一部分,作为数据测试使用
TestData : {3.1,5.9,9.5
            3.3,5.9,10.2
            3.5,6.3,10.9
            3.7,6.7,11.6
            3.9,7.1,12.3}
'''
import numpy as np
import random
from numpy import genfromtxt

#在表中取数据测试集
def getData(dataSet):
    #读书表格的维度
    m, n = np.shape(dataSet)
    trainData = np.ones((m, n))
    trainData[:,:-1] = dataSet[:,:-1]
    trainLabel = dataSet[:,-1]
    return trainData, trainLabel

def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    xTrains = x.transpose()
    for i in range(0, maxIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        # print loss
        gradient = np.dot(xTrains, loss) / m
        theta = theta - alpha * gradient
    return theta

def predict(x, theta):
    m, n = np.shape(x)
    xTest = np.ones((m, n+1))
    xTest[:, :-1] = x
    yP = np.dot(xTest, theta)
    return yP
#1.首先将数据读入Python中
dataPath = r"house.csv"
dataSet = genfromtxt(dataPath, delimiter=',')

#2.接下来将读取的数据分别得到自变量矩阵和因变量矩阵：
trainData, trainLabel = getData(dataSet)
m, n = np.shape(trainData)

#3.初始化权重值  ,学习率  最大迭代次数
theta = np.ones(n)
alpha = 0.1
maxIteration = 5000

#4.开始计算
theta = batchGradientDescent(trainData, trainLabel, theta, alpha, m, maxIteration)

#5.导入测试数据集
x = np.array([[3.1, 5.5], [3.3, 5.9], [3.5, 6.3], [3.7, 6.7], [3.9, 7.1]])

#6.开始预测
print (predict(x, theta))