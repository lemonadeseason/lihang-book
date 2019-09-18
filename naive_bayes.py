#naive bayes to solve example in p50

import numpy as np

#X_train、Y_train分别是训练集的所有特征向量和对应标签，每一个样本的特征向量放在X_train的对应list中
#X_train = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'],  [1, 'S'],[2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'],  [2, 'L'],[3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'],  [3, 'L']]
#Y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1] 

#训练模型的函数，返回先验概率P(Y=y)以及条件概率P（X1i|Y）,即当y取特定值时，第几个特征取某个值的概率
def Train(X_train,Y_train):
    Y_value = [1,-1]
    X_value = [[1,2,3],['S','M','L']]
    prior_probability = np.zeros(2)
    conditional_probability = np.zeros((2,2,3))  #y有两种取值，x的特征2维，最大取值种数3
    positive_count = negative_count = 0 #统计先验概率
    for i in range(len(Y_train)):
        if(Y_train[i] == 1):
            positive_count += 1
        else:
            negative_count += 1
    prior_probability[0] = (float(positive_count))/len(Y_train)
    prior_probability[1] = (float(negative_count))/len(Y_train)
    for i in range(len(Y_train)):
        for j in range(2):    #对样本的每一维特征都要统计
            index1 = Y_value.index(Y_train[i])
            index2 = j
            index3 = X_value[j].index(X_train[i][j])
            conditional_probability[index1][index2][index3] += 1
    conditional_probability[0] /= positive_count
    conditional_probability[1] /= negative_count
    return prior_probability,conditional_probability
    
def Predict(test,prior_probability,conditional_probabilty):
    #test是测试，例如[2,S],应该返回一个list，代表后验概率，P（Y|X）
    Y_value = [1,-1]
    X_value = [[1,2,3],['S','M','L']]
   
    #特征的下标
    index1 = X_value[0].index(test[0])
    index2 = X_value[1].index(test[1])
    
    result = np.zeros(2)
    for i in range(2):
        result[i] = prior_probability[i]*conditional_probabilty[i][0][index1]*conditional_probabilty[i][1][index2]
    return result
    
X_train = [[1, 'S'], [1, 'M'], [1, 'M'], [1, 'S'],  [1, 'S'],[2, 'S'], [2, 'M'], [2, 'M'], [2, 'L'],  [2, 'L'],[3, 'L'], [3, 'M'], [3, 'M'], [3, 'L'],  [3, 'L']]
Y_train = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1] 
prior_probability,conditional_probabilty = Train(X_train,Y_train)
print(Predict([2,'S'],prior_probability,conditional_probabilty))
