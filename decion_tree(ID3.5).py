#这个代码基本对于https://github.com/fengdu78/lihang-code/blob/master/%E7%AC%AC05%E7%AB%A0%20%E5%86%B3%E7%AD%96%E6%A0%91/5.DecisonTree.ipynb
#未作改动，原代码很优美，无论是对pandas的操作，还是在决策树和节点的理解上都很值得一读再读
#在node这个类中，保存了类别label、划分属性feature_name，最具有启发意义的是保存了一个命名为tree的字典，在其中保存了在feature_name这个属性上取值
#不同的数据集（也是树），并且实现了类的repr方法，最终输出答案。还有在node中保存result、DTree中保存tree_，并用fit方法得到整个根节点

#唯一改变的是删除了node中如feature、root之类的属性（暂时看起来并没有什么用），在train函数中生成node时也对应修改了。
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from math import log
import pprint

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄',u'有工作',u'有自己的房子',u'信贷情况',u'类别']
    return datasets,labels

class Node:
    def __init__(self，label=None,feature_name=None):
        #self.root = root
        self.label = label    #叶节点的话，代表类别     string
        self.feature_name = feature_name    #特征名，string
        #self.feature = feature               #feature是非叶节点中的，代表用来spilt的节点在当时数据集中在第几列    int
        self.tree = {}    #用来保存这个特征的划分情况，是一个字典，key是这个特征的某个取值，对应value是特征为这个取值的对应树
        self.result = {
            'label:':self.label,  
            'spilt_feature_name:':self.feature_name,
            'tree:':self.tree
        }
        
    def __repr__(self):    #print node类型的变量时，得到以下字符串
        return '{}'.format(self.result)
    def add_node(self,val,node):
        self.tree[val] = node
        
class DTree:
    def __init__(self,epsilon=0.1):
        self.epsilon = epsilon
        #self._tree = {}                    #不知道这里为什么是字典，明明是Node类型的变量啊... 
        self._tree = Node()                      
    
    #熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)    #样本总数
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum(p/data_length*log(p/data_length,2)
                   for p in label_count.values())
        return ent
    
    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p)
                        for p in feature_sets.values()])
        return cond_ent
    
    # 信息增益
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_
    
    def train(self,train_data):
        
        y_train,features = train_data.iloc[:,-1],train_data.columns[:-1]
        #1.若样本都属于同一个类别，则结束，并且用这个类别作为节点的类标记
        if len(y_train.value_counts())==1:
            return Node(label=y_train.iloc[0])    #根？
        
        #2.如果没有用来分类的特征(之前用来分类的是仅剩的最后一个特征了，现在只能用类别中占大多数的作为这个节点的类别)
        if len(features)==0:
            return Node(label=y_train.value_counts().sort_values(ascending=False).index[0])
        
        #3.计算各特征对数据集的信息增益
        #需要np.array吗？
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]
        
        #4.如果最大信息增益小于阈值，则不再继续分类
        if max_info_gain<self.epsilon:
            return Node(label=y_train.value_counts().sort_values(ascending=False).index[0])
        
        #5.否则，继续用带来最大信息增益的特征对样本分类
        node_tree = Node(feature_name=max_feature_name)
        
        feature_list = train_data[max_feature_name].value_counts().index #这一特征在样本集上所有可能的取值
        for f in feature_list:
            #这里的pandas的index语法不太明白
            sub_train_df = train_data.loc[train_data[max_feature_name]==f].drop([max_feature_name],axis=1)
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f,sub_tree)      
            
        return node_tree
    
    def fit(self,train_data):
        self._tree = self.train(train_data)
        return self._tree
    
datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)

#print tree
#预期输出：{'label:': None, 'spilt_feature_name:': '有自己的房子', 'tree:': {'否': {'label:': None, 'spilt_feature_name:': '有工作', 'tree:': {'否': {'label:': '否', 'spilt_feature_name:': None, 'tree:': {}}, '是': {'label:': '是', 'spilt_feature_name:': None, 'tree:': {}}}}, '是': {'label:': '是', 'spilt_feature_name:': None, 'tree:': {}}}}
