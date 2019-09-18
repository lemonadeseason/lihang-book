# lihang-book
李航《统计学习方法》的实现，希望可以提升对机器学习的认识（以前都是半途而废或者不求甚解），还有拯救弱鸡编码能力。

感知机（Perceptron）：

遗留问题：
1.为什么L（w，b）可以不管L2范数？       
2.为什么可以每次更新使用一个点，只是随机梯度下降的直接应用吗？

基本思路：
用于解决线性可分（存在超平面s，对于所有正样本有wx+b>0，对于所有反样本有wx+b<0）的问题。目标是要找到这样的一个超平面，待定需求解的是w（权重）、b（偏置）。模型的问题解决了，接下来是算法，即定义损失函数（最好对于待求解参数连续可导，便于优化），这样的函数可以是：所有误分类点到超平面的函数距离和。

实现：
把书上例2.1用算法2.1解决的方法代码化了，参考https://blog.csdn.net/SoyCoder/article/details/82946917 ，先看懂之后自己敲了一遍，numpy和python中函数语法、matplotlib很多东西都需要巩固。

朴素贝叶斯（naive bayes）：
实现了书上的例4.1，问题有：需要在train和predict中手动指定特征个数，每个特征可选值的个数，且没有对应关系，如哪个特征对应哪个概率，应该需要一个map之类的数据结构来对应。以后可以改进，来扩大代码的适用范围。

