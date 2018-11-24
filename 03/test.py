# -*- coding: utf-8 -*-
# @Time    : 2018/8/28 11:27
# @Author  : DONG
# @File    : test.py
# @Software: PyCharm

import numpy as np

# A是array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])
A = np.arange(16)

# 将A变换为三维矩阵
A = A.reshape(2,2,4)
print(A)

print(A.transpose((1,0,2))[0][1][2])
print(A.transpose((0,1,2))[0][1][2])