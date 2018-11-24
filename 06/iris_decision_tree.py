import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # 对训练数据进行分割
from sklearn.metrics import accuracy_score   # 评估准确率
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()  #载入数据
data = pd.DataFrame(iris.data)   #把数据变成数据框的类型 ,易于操作和查看
data.columns = iris.feature_names   #特征名
data['Species'] = load_iris().target   #提取target数据,给data,并且重新命名一列的名为Species , 花的种类
# print(data)

x = data.iloc[:, :2]  # 花萼长度和宽度
y = data.iloc[:, -1]
# y = pd.Categorical(data[4]).codes
# print(x)
# print(y)

#训练数据,把数据分成训练数据 和 测试数据  random_state 随机种子
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)
# criterion  分割标准 , 默认的是gini  , max_depth 最大的深度,就是几层 , 默认为空,不限制
tree_clf = DecisionTreeClassifier(max_depth=8, criterion='entropy')
tree_clf.fit(x_train, y_train)
y_test_hat = tree_clf.predict(x_test)
#判断 测试值和正确值的准备率
print("acc score:", accuracy_score(y_test, y_test_hat))

"""
export_graphviz(
    tree_clf,
    out_file="./iris_tree.dot",
    feature_names=iris.feature_names[:2],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# ./dot -Tpng ~/PycharmProjects/mlstudy/bjsxt/iris_tree.dot -o ~/PycharmProjects/mlstudy/bjsxt/iris_tree.png
"""
# 这个带入花萼的长度和宽度 ,得到的是预测值 , 不是0 就是1  ,有三个  [0 , 1 ,0] ,选择最大的,就是1
print(tree_clf.predict_proba([[5, 1.5]]))
# 这里返回的是预测值被选中的值所在的位置, 是1 ,因为上面的位置是0 1 2,
print(tree_clf.predict([[5, 1.5]]))

#上面max_depth 每次都是自己修改大小,达到最优的,这里就是优化代码,让代码自己寻找最优
depth = np.arange(1, 15)
err_list = []
for d in depth:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
    clf.fit(x_train, y_train)
    y_test_hat = clf.predict(x_test)
    result = (y_test_hat == y_test)
    if d == 1:
        print(result)
    err = 1 - np.mean(result)
    print(100 * err)
    err_list.append(err)
    print(d, ' 错误率：%.2f%%' % (100 * err))

#设置字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
#图的底色是白色
plt.figure(facecolor='w')
# 画图  ro-  红色  点是○  每个人用-连接起来  lw=2 线的宽度   linewidth
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('决策树深度', fontsize=15)
plt.ylabel('错误率', fontsize=15)
plt.title('决策树深度和过拟合', fontsize=18)
plt.grid(True)
plt.show()


# tree_reg = DecisionTreeRegressor(max_depth=2)
# tree_reg.fit(X, y)

