#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_0.py

import numpy as np
import matplotlib.pyplot as plt



# 这里相当于是随机X维度X1，rand是随机均匀分布,生成100行,每行一个数据的列向量
X = 2 * np.random.rand(100, 1)
# 人为的设置真实的Y一列，np.random.randn(100, 1)是设置error，randn是标准正太分布
#randn 生成标准正态分布的伪随机数（均值为0，方差为1）
y = 4 + 3 * X + np.random.randn(100, 1)
# 整合X0和X1 ,c_代表  compile ,整合 ,代表参数w0...wn的初始化
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)

# 常规等式求解theta  ,linalg 线性代数
#np.linalg.inv(a)    #矩阵a的逆矩阵  np.dot(a,b)用来计算数组的点积
#dot_ = x.dot(y)  # 等价于np.dot(x, y)
#mat = X.T.dot(X)   # X.T x的转置 与 x 的 dot内积
# 常规等式求解theta =[X^T · X ] · X^T ·Y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# y = w0 + w1*x 求出两个    参数 w0 w1
print(theta_best)
'''
结果 : 
[ [4.07410678] 
  [2.93191986] ]
'''
# 创建测试集里面的X1 , 输出 array[ [0] , [2] ]
X_new = np.array([[0], [2]])
print(X_new)
X_new_b = np.c_[(np.ones((2, 1))), X_new]
print(X_new_b)
y_predict = X_new_b.dot(theta_best)
print(y_predict)

#画图
plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
#控制x  y 的坐标轴范围
plt.axis([0, 2, 0, 15])
#显示
plt.show()



