# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:18:59 2017

@author: jk
"""

import tensorflow as tf

import numpy as np



#第一部分随机生成100点（x，y）数据input

x_data = np.random.rand(100).astype(np.float32)

y_data = x_data * 0.1 + 0.3

#第二部分创建权重和偏移量
"""
构建线性模型的tensor变量W, btf.random_uniform([1], -1.0, 1.0)：构建一个tensor, 
该tensor的shape为[1]，该值符合[-1, 1)的均匀分布。其中[1]表示一维数组，里面包含1个元素。
tf.Variable(initial_value=None)：构建一个新变量，该变量会加入到TensorFlow框架中的图集合中。
tf.zeros([1])：构建一个tensor, 该tensor的shape为[1], 里面所有元素为0。

"""
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

b = tf.Variable(tf.zeros([1]))

y = W * x_data + b

# 第三部分建立损失方程和优化器
"""
#构建损失方程，优化器及训练模型操作train
tf.square(x, name=None)：计算tensor的平方值。
tf.reduce_mean(input_tensor)：计算input_tensor中所有元素的均值。
tf.train.GradientDescentOptimizer(0.5)：构建一个梯度下降优化器，0.5为学习速率。
学习率决定我们迈向（局部）最小值时每一步的步长，设置的太小，那么下降速度会很慢，
设的太大可能出现直接越过最小值的现象。所以一般调到目标函数的值在减小而且速度适中的情况。
optimizer.minimize(loss)：构建一个优化算子操作。使用梯度下降法计算损失方程的最小值。
loss为需要被优化的损失方程。
"""
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


#第四部分构建变量初始化操作init
"""
#tf.initialize_all_variables()：初始化所有TensorFlow的变量。
"""
init = tf.initialize_all_variables()
"""
tf.Session()：创建一个TensorFlow的session，在该session种会运行TensorFlow的图计算模型。
"""
sess = tf.Session()
"""
初始化所有TensorFlow变量sess.run()：在session中执行图模型的运算操作。
如果参数为tensor时，可以用来求tensor的值。
"""
sess.run(init)



#第五部分训练该线性模型，每隔20次迭代，输出模型参数

for step in range(201):

    sess.run(train)

    if step % 20 == 0:

        print(step, sess.run(W), sess.run(b))