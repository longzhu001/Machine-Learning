#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: insurance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


'''
  read_csv  读取一个 csv文本文件
  data.head() 查看前五行数据  
  data.tail() 查看最后五行数据
  data.describe()  统计摘要 , 做数据的百分比
  iloc[:n]    可以解决选择多少行
'''
data = pd.read_csv('./insurance.csv')
print(type(data))
print(data.head())
print(data.tail())
print(data.describe())
'''
    # 采样要均匀
    @:param value_counts()  查看这一列,去重之后,有多少种数据     
    @:param corr()  列和列之间的相关性,利用是的Pearson(皮尔逊)相关系数(前提只能是数字数据)
        对于 自变量 和 自变量 之间的P
        P得出的结果 -1 <--> 1   -1 的是负相关  1 是正相关  0 不相关  ,值都是接近
        正相关,代表两个变量想类似,可以合并项或者去掉,达到降低维度的效果
        对于 因变量  和  自变量  之间的P
        若是正相关,证明他们之间的相关性好, 会因为自变量的变化,而导致因变量的大幅度变化
'''
data_count = data['age'].value_counts()
print(data_count)
# data_count[:10].plot(kind='bar')
# plt.show()
# plt.savefig('./temp')
"""
print(data.corr())

reg = LinearRegression()
x = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y = data['charges']
# python3.6 报错 sklearn ValueError: could not convert string to float: 'northwest'，加入一下几行解决
x = x.apply(pd.to_numeric, errors='coerce')    #把相对应的问题转成数字
y = y.apply(pd.to_numeric, errors='coerce')    
x.fillna(0, inplace=True)                      #若是数据为空置,默认为 0
y.fillna(0, inplace=True)

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(x)

reg.fit(X_poly, y)
print(reg.coef_)
print(reg.intercept_)

y_predict = reg.predict(X_poly)

plt.plot(x['age'], y, 'b.')
plt.plot(X_poly[:, 0], y_predict, 'r.')
plt.show()
"""