#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: linear_regression_1.py
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
和01.py  的功能是一样de,都是构建线性回归方程
'''

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

#regr为回归过程，fit(x,y)进行回归
#regr = LinearRegression().fit(dataSet_x, dataSet_y)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
#lin_reg.intercept_   截距 , lin_reg.coef_  参数
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.array([[0], [2]])
#用predic预测，这里预测输入x对应的值，进行画线
print(lin_reg.predict(X_new))
#真实值的散点图,scatter 就是绘制散点图的
plt.scatter(X, y,  color='black')
#预测直线的图
plt.plot(X_new, lin_reg.predict(X_new), color='red', linewidth=1)
plt.show()






