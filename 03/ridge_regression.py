#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: ridge_regression.py

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

"""
#创建一个Ridge实例 , solver='sag' 是你选择梯度的方式 , sag == SGD
ridge_reg = Ridge(alpha=1, solver='sag')
#fit回归分析 , 创建对应的模型
ridge_reg.fit(X, y)
#预测 1.5 的预测值是多少
print(ridge_reg.predict(1.5))
#截距
print(ridge_reg.intercept_)
#参数
print(ridge_reg.coef_)
"""

#创建一个SGDRegressor的实例,
sgd_reg = SGDRegressor(penalty='l2', n_iter=1000)
#更改数组的形状 ,把数组上的每一行数据,抽取出来,变换成一行多列
sgd_reg.fit(X, y.ravel())
#预测
print(sgd_reg.predict(1.5))
#截距
print("W0=", sgd_reg.intercept_)
#参数
print("W1=", sgd_reg.coef_)

