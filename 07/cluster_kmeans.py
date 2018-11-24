# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

'''
MiniBatchKMeans :
当数据量很大的时候，Kmeans 显然还是很弱的，会比较耗费内存速度也会收到很大影响。scikit-learn 提供了MiniBatchKMeans算法，大致思想就是对数据进行抽样，每次不使用所有的数据来计算，这就会导致准确率的损失。

kmeans : 
    优点：
原理简单
速度快
对大数据集有比较好的伸缩性
    缺点：
需要指定聚类 数量K
对异常值敏感
对初始值敏感
'''
def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    #创建400个样本
    N = 400
    #四个类别
    centers = 4
    # 创建 400 个聚类的模拟数据 , 返回数据 和 数据的长度
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 2.5, 0.5, 2), random_state=2)
    '''
    #函数原型：vstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组
     它是垂直（按照行顺序）的把数组给堆叠起来
    #函数原型：hstack(tup) ，参数tup可以是元组，列表，或者numpy数组，返回结果为numpy的数组。
     它其实就是水平(按列顺序)把数组给堆叠起来，vstack()函数正好和它相反。
    '''
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)

    cls = KMeans(n_clusters=4, init='k-means++')
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)
    # 二维数组
    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)
    y_r_hat = cls.fit_predict(data_r)

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9, 10), facecolor='w')
    plt.subplot(421)
    plt.title(u'原始数据')
    plt.scatter(data[:, 0], data[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data, axis=0)
    x1_max, x2_max = np.max(data, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title(u'KMeans++聚类')
    plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(423)
    plt.title(u'旋转后数据')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(424)
    plt.title(u'旋转后KMeans++聚类')
    plt.scatter(data_r[:, 0], data_r[:, 1], c=y_r_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title(u'方差不相等数据')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2, s=30, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data2, axis=0)
    x1_max, x2_max = np.max(data2, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(426)
    plt.title(u'方差不相等KMeans++聚类')
    plt.scatter(data2[:, 0], data2[:, 1], c=y2_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(427)
    plt.title(u'数量不相等数据')
    plt.scatter(data3[:, 0], data3[:, 1], s=30, c=y3, cmap=cm, edgecolors='none')
    x1_min, x2_min = np.min(data3, axis=0)
    x1_max, x2_max = np.max(data3, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.subplot(428)
    plt.title(u'数量不相等KMeans++聚类')
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(True)

    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle(u'数据分布对KMeans聚类的影响', fontsize=18)
    # https://github.com/matplotlib/matplotlib/issues/829
    # plt.subplots_adjust(top=0.92)
    plt.show()
    # plt.savefig('cluster_kmeans')


