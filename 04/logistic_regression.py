#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: logistic_regression.py

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

# datasets.load_iris() 自带模块中的数据集,数据集中有一个简单的数据提供iris
iris = datasets.load_iris()
# 取出数据集中的键值对中的键,看有什么方法可以调用
print(list(iris.keys()))
# 数据集的描述信息
print(iris['DESCR'])
# 数据集中数据的特征名
print(iris['feature_names'])

# 花瓣的宽度
X = iris['data'][:, 3:]
print(X)

# 目标数据
print('目标数据:',iris['target'])
y = iris['target']
# y = (iris['target'] == 2).astype(np.int)
print(y)


# Utility function to report best scores
# def report(results, n_top=3):
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")
#
#
# start = time()
# param_grid = {"tol": [1e-4, 1e-3, 1e-2],
#               "C": [0.4, 0.6, 0.8]}
log_reg = LogisticRegression(multi_class='ovr', solver='sag')
# grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3)
log_reg.fit(X, y)
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)

'''
numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None) 
功能：生成一个指定间隔数的数组。 
参数：（1）start：序列的起始值（2）stop：序列的终止值（3）生成的样本数（默认值=50） 
（4）endpoint：该值取True时，序列的终止值包括stop；反之不包括。 
（5）retstep：该值取True时，生成的数组中显示间距；反正不显示。 
（6）dtype：数据类型；输出数组的类型。如果没有给出dtype，则从其他输入参数中推断数据类型。
  reshape(-1, 1)  生成明确的行和列,其中-1 代表最大的行 ,1 代表生成一列 ,构成有序的数组
'''
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)

'''
# 返回预测标签  
print(log_reg.predict(x_test))  
# 返回预测属于某标签的概率  
print(log_reg.predict_proba(x_test)) 
'''
y_proba = log_reg.predict_proba(X_new)
y_hat = log_reg.predict(X_new)
print(y_proba)
print(y_hat)

plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Iris-Setosa')
plt.show()

print(log_reg.predict([[1.7], [1.5]]))


