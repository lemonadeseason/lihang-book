#p29页，例2.1用算法2.1解决，将数据和标签分别存放。另外在Perception类中写初始化函数，训练模型函数，以及将最终得到的超平面和样本绘制出来。
#如果在jupyter中运行，需要“%matplotlib inline”。


import numpy as np
import matplotlib.pyplot as plt

input_data = np.array([[3,3],[4,3],[1,1]])
input_label = np.array([1,1,-1])

def sign_func(x):
   if x>0:
      return 1
   else:
      return -1

class Perceptron():
   def __init__(self):
      self.weight = [0,0]
      self.bias = 0     
      self.delta = 1      #learning rate

   def train(self):
      for i in range(100):    #这个例子肯定用不了100次迭代就可以找到超平面，但是更好的做法是用一个变量表示是否在这轮循环中发现了错分类的点。
         count = -1
         for j in range(len(input_data)):
            if input_label[j] != sign_func(np.dot(input_data[j],self.weight)+self.bias):
               count = j      #j th sample will be mislabled if perceptron not corrected
               break
         if count != -1:
            self.weight = self.weight + self.delta*input_data[count]*input_label[count]
            self.bias = self.bias + self.delta*input_label[count]
   
   def plot(self):
        plt.figure()
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        x_line = [0, 10]

        y_line = [0, 0]

        for i in range(len(x_line)):

            y_line[i] = -(x_line[i] * self.weight[0] + self.bias) / self.weight[1]

        plt.plot(x_line, y_line)

        for index in range(len(input_data)):

            if input_label[index] == 1:

                plt.plot(input_data[index][0], input_data[index][1], 'bo')

            else:

                plt.plot(input_data[index][0], input_data[index][1], 'ro')

        plt.show()

if __name__ == '__main__':
   p = Perceptron()
   p.train()
   p.plot()
