import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#参考代码中使用的是转换为了numpy中的array形式，不知道下面这样对不对
data = np.array(df.iloc[:100,[0,1,-1]])    #只取0,1两种类别的样本，只用前两个特征
X,y = data[:,:-1],data[:,-1]   #特征，标签
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)   #从全集中划分训练，测试（基本操作）

#knn类，用户可以指定不同的k值，以及距离度量中的p值
class KNN:
    def __init__(self,X_train,y_train,k=3,p=2):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p
    
    def predict(self,X):
        #遍历所有的点，找到距离X最近的k个点，少数服从多数
        #维护一个knn_list，是一个list，每个元素是一个tumple，第一元是距离，第二元是对应的label
        knn_list = []
        for i in range(self.k):
            dist = np.linalg.norm(X_train[i]-X,ord=self.p)   #求范数的函数
            knn_list.append((dist,y_train[i]))
        for i in range(self.k,len(X_train)):
            max_index = knn_list.index(max(knn_list,key=lambda x:x[0]))    #巧妙利用python的语法
            dist = np.linalg.norm(X_train[i]-X,ord=self.p)
            if dist<knn_list[max_index][0]:
                knn_list[max_index] = (dist,y_train[i])
        knn = [k[-1] for k in knn_list]    #很巧妙
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(),key=lambda x:x[1])[-1][0]
        return max_count
    
    
    
    def score(self,X_test,y_test):
        right_count = 0
        for i in range(len(X_test)):
            if self.predict(X_test[i]) == y_test[i]:
                right_count += 1
        return right_count/len(X_test)
        
 clf = KNN(X_train,y_train)
 clf.score(X_test,y_test)
