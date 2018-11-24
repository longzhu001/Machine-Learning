from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, :2]  # 花萼长度和宽度
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# 随机森林,森林是多大, n_estimators 15颗树构成
# n_jobs 指的是你有多少个线程
rnd_clf = RandomForestClassifier(n_estimators=15, max_leaf_nodes=16, n_jobs=1)
rnd_clf.fit(X_train, y_train)

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
    n_estimators=15, max_samples=1.0, bootstrap=True, n_jobs=1
)
bag_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
y_pred_bag = bag_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))
print(accuracy_score(y_test, y_pred_bag))


# Feature Importance ,特征的重要性
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rnd_clf.fit(iris["data"], iris['target'])
for name, score in zip(iris['feature_names'], rnd_clf.feature_importances_):
    print(name, score)
'''
上面的是第三种选中特征的重要性方法了 ! 
   第一种 :  皮尔逊persion,相关性系数
   第二种 : L1 正则  lasso , 其中会把部分w趋近0  w->0 , 而趋近1的w->1 , 就是重要性明显的
   第三种 : 就是上面的,由随机森林来做的 RandomForestClassifier
'''






