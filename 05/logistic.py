# coding:utf-8

import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import fft
from scipy.io import wavfile

'''
作用  : 音乐分类器  , 但是只能适用单声道
步骤  :
        前提 : 分类好的文件夹  归类的级别   归类好之后再去读取文件  
        1.读取构建的的多种类别音乐,其中每种类别的音乐的采集样本都必须相同,防止过拟合
        2.读取之后,同过 fft 傅里叶变换函数,进行解析,取前1000个音频,解析会返回两个参数
            参数  :   采样率    音乐本身  由于只是取了前1000个,所以音乐每个的大小都是固定的
        3.通过傅里叶变换转化成音频了之后,我们就可以利用计算机来识别
        4.最后通过 numpy中的save 函数来保存 音乐本身  这个参数  和 音乐属于的类别
        5.加载训练集数据,分割训练集以及测试集,进行分类器的训练----构造训练集
        6.开始读取由1--5步骤生成的npy文件,进行np.load 训练, 生成的对象用列表X保存 
        7.列表保存之后,构建所属类别的Y列表,由于skl 无法识别正常的python列表,需要把列表转成np.array 数组
        8.开始 建立 linear 模型,构建模型对象model
        9.利用模型,model.fix函数 ,训练 X Y  参数,  得到系数
        10. 最后进行预测,得到6个概率,因为是六个分类,取最大的概率的,返回的是列表中的某一列数字
        11.再返回列表中,利用某一列的数字,提取列表中的具体对应的参数,完成整个分类模型

'''
"""
n = 40
# hstack使得十足拼接
# rvs是Random Variates随机变量的意思
# 在模拟X的时候使用了两个正态分布,分别制定各自的均值,方差,生成40个点
X = np.hstack((norm.rvs(loc=2, size=n, scale=2), norm.rvs(loc=8, size=n, scale=3)))
# zeros使得数据点生成40个0,ones使得数据点生成40个1
y = np.hstack((np.zeros(n),np.ones(n)))
# 创建一个 10 * 4 点（point）的图，并设置分辨率为 80
plt.figure(figsize=(10, 4),dpi=80)
# 设置横轴的上下限
plt.xlim((-5, 20))
# scatter散点图
plt.scatter(X, y, c=y)
plt.xlabel("feature value")
plt.ylabel("class")
plt.grid(True, linestyle='-', color='0.75')
plt.savefig("D:/workspace/scikit-learn/logistic_classify.png", bbox_inches="tight")
"""

"""
# linspace是在-5到15的区间内找10个数
xs=np.linspace(-5,15,10)

#---linear regression----------
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
# reshape重新把array变成了80行1列二维数组,符合机器学习多维线性回归格式
clf.fit(X.reshape(n * 2, 1), y)
def lin_model(clf, X):
    return clf.intercept_ + clf.coef_ * X

#---logistic regression--------
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression()
# reshape重新把array变成了80行1列二维数组,符合机器学习多维线性回归格式
logclf.fit(X.reshape(n * 2, 1), y)
def lr_model(clf, X):
    return 1.0 / (1.0 + np.exp(-(clf.intercept_ + clf.coef_ * X)))

#----plot---------------------------    
plt.figure(figsize=(10, 5))
# 创建一个一行两列子图的图像中第一个图
plt.subplot(1, 2, 1)
plt.scatter(X, y, c=y)
plt.plot(X, lin_model(clf, X),"o",color="orange")
plt.plot(xs, lin_model(clf, xs),"-",color="green")
plt.xlabel("feature value")
plt.ylabel("class")
plt.title("linear fit")
plt.grid(True, linestyle='-', color='0.75')
# 创建一个一行两列子图的图像中第二个图
plt.subplot(1, 2, 2)
plt.scatter(X, y, c=y)
plt.plot(X, lr_model(logclf, X).ravel(),"o",color="c")
plt.plot(xs, lr_model(logclf, xs).ravel(),"-",color="green")
plt.xlabel("feature value")
plt.ylabel("class")
plt.title("logistic fit")
plt.grid(True, linestyle='-', color='0.75')

plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)     
plt.savefig("D:/workspace/scikit-learn/logistic_classify2.png", bbox_inches="tight")
"""


"""
使用logistic regression处理音乐数据,音乐数据训练样本的获得和使用快速傅里叶变换（FFT）预处理的方法需要事先准备好
1. 把训练集扩大到每类100个首歌而不是之前的10首歌,类别仍然是六类:jazz,classical,country, pop, rock, metal
2. 同时使用logistic回归和KNN作为分类器
3. 引入一些评价的标准来比较Logistic和KNN在测试集上的表现 
"""



'''
# 准备音乐数据   通过  np  转换成计算机可以识别的东西
    str(n).zfill(5)  拼接一个字符串,是5位的,如果不够五位,用0代表,zero 
    np.save  保存的文件,自带后缀,  npy
    wavfile.read  返回的参数有两个 一个是   采样率    一个是 音乐本身
    fft(X)   把音乐文件进行傅里叶变换
'''
def create_fft(g, n):
    rad = "d:/genres/"+g+"/converted/"+g+"."+str(n).zfill(5)+".au.wav"
    sample_rate, X = wavfile.read(rad)
    fft_features = abs(fft(X)[:1000])
    sad = "d:/trainset/"+g+"."+str(n).zfill(5) + ".fft"
    np.save(sad, fft_features)
    
# -------create fft--------------

# 六分类
genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
for g in genre_list:
    for n in range(100):
        create_fft(g, n)


# 加载训练集数据,分割训练集以及测试集,进行分类器的训练
# 构造训练集！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# -------read fft--------------
genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
X = []
Y = []
for g in genre_list:
    for n in range(100):
        rad = "D:/PythonProject/Machine-Learning/05/trainset/"+g+"."+str(n).zfill(5)+ ".fft"+".npy"
        fft_features = np.load(rad)
        X.append(fft_features)
        Y.append(genre_list.index(g))
# skl 无法正常识别列表, 需要改成np中的数组才可以进行识别
X = np.array(X)
Y = np.array(Y)
"""
# 首先我们要将原始数据分为训练集和测试集，这里是随机抽样80%做测试集，剩下20%做训练集 
import random
randomIndex=random.sample(range(len(Y)),int(len(Y)*8/10))
trainX=[];trainY=[];testX=[];testY=[]
for i in range(len(Y)):
    if i in randomIndex:
        trainX.append(X[i])
        trainY.append(Y[i])
    else:
        testX.append(X[i])
        testY.append(Y[i])
"""
        
# 接下来，我们使用sklearn，来构造和训练我们的两种分类器 
# ------train logistic classifier--------------
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, Y)
# predictYlogistic=map(lambda x:logclf.predict(x)[0],testX)

# 可以采用Python内建的持久性模型 pickle 来保存scikit的模型
"""
>>> import pickle
>>> s = pickle.dumps(clf)
>>> clf2 = pickle.loads(s)
>>> clf2.predict(X[0])
"""

"""
#----train knn classifier-----------------------
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(trainX) 
predictYknn=map(lambda x:trainY[neigh.kneighbors(x,return_distance=False)[0][0]],testX)

# 将predictYlogistic以及predictYknn与testY对比，我们就可以知道两者的判定正确率 
a = np.array(predictYlogistic)-np.array(testY)
print a, np.count_nonzero(a), len(a)
accuracyLogistic = 1-np.count_nonzero(a)/(len(a)*1.0)
b = np.array(predictYknn)-np.array(testY)
print b, np.count_nonzero(b), len(b)
accuracyKNN = 1-np.count_nonzero(b)/(len(b)*1.0)

print "%f" % (accuracyLogistic)
print "%f" % (accuracyKNN)
"""

print('Starting read wavfile...')
# prepare test data-------------------
# sample_rate, test = wavfile.read("d:/trainset/sample/outfile.wav")
#D:\PythonProject\Machine-Learning\05\trainset\sample\heibao-wudizirong-remix.wav
sample_rate, test = wavfile.read("d:/StudyMaterials/python/python-sklearn/trainset/sample/heibao-wudizirong-remix.wav")
# sample_rate, test = wavfile.read("d:/genres/metal/converted/metal.00080.au.wav")
testdata_fft_features = abs(fft(test))[:1000]
print(sample_rate, testdata_fft_features, len(testdata_fft_features))
# 用模型预测,预测之后,取最大的那一个
type_index = model.predict([testdata_fft_features])[0]
print(type_index)
print(genre_list[type_index])

"""
from sklearn.metrics import confusion_matrix
cmlogistic = confusion_matrix(testY, predictYlogistic)
cmknn = confusion_matrix(testY, predictYknn)

def plotCM(cm,title,colorbarOn,givenAX):
    ncm=cm/cm.max()
    plt.matshow(ncm, fignum=False, cmap='Blues', vmin=0, vmax=2.0)
    if givenAX=="":
        ax=plt.axes()
    else:
        ax = givenAX
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    plt.title(title,size=12)
    if colorbarOn=="on":
        plt.colorbar()
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(i,j,cm[i,j],size=15)

plt.figure(figsize=(10, 5))  
fig1=plt.subplot(1, 2, 1)          
plotCM(cmlogistic,"confusion matrix: FFT based logistic classifier","off",fig1.axes)   
fig2=plt.subplot(1, 2, 2)     
plotCM(cmknn,"confusion matrix: FFT based KNN classifier","off",fig2.axes) 
plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)     

plt.savefig("d:/confusion_matrix.png", bbox_inches="tight")
"""