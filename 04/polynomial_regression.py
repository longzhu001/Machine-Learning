#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: polynomial_regression.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.plot(X, y, 'b.')

d = {1: 'g-', 2: 'r+', 10: 'y*'}


for i in d:
    ''' 
        #多元回归
        通过 PolynomialFeatures() 类自动产生多项式特征矩阵
        @:param  degree: 阶数  多项式次数，默认为 2 次多项式
        @:param  include_bias: 默认为 True，包含多项式中的截距项

    '''
    poly_features = PolynomialFeatures(degree=i, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    print(X[0])
    print(X_poly[0])
    print(X_poly[:, 0])

    '''
       #线性回归
       @:param fit_intercept ,就是回归自动把w0 , 截距的部分加上
       @:param lin_reg.intercept_ 截距
       @:param lin_reg.coef_ 参数
       @:param lin_reg.predict  预测
    '''
    lin_reg = LinearRegression(fit_intercept=True)
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)
    #
    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])

plt.show()
