#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名: batch_gradient_descent.py
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)

learning_rate = 0.1
#迭代次数,无论是否是最优解,只要迭代够了,就1000次后的解,是最优的,迭代次数越多,越准确
n_iterations = 10000
m = 100

# 1，初始化theta，w0...wn
theta = np.random.randn(2, 1)
count = 0

# 4，不会设置阈值，之间设置超参数，迭代次数，迭代次数到了，我们就认为收敛了
for iteration in range(n_iterations):
    count += 1
    # 2，接着求梯度gradient 1/m * X^T · (X_b.dot(theta)-y  估计值 y - 实际值 y )
    gradients = 1/m * X_b.T.dot(X_b.dot(theta)-y)
    # 3，应用公式调整theta值，theta_t + 1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients

print(count)
print(theta)









