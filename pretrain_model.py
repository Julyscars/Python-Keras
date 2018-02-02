# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:48:12 2017

@author: Atlantis
"""

import os
import numpy as np
from tqdm import tqdm
import jieba
from time import time
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle


#words, embeddings = pickle.load(open('polyglot-zh.pkl', 'rb'), encoding='latin1', errors='strict')
#print("Emebddings shape is {}".format(embeddings.shape))

#word_vectors = KeyedVectors.load_word2vec_format('zh_wiki/zhwiki_2017_03.sg_50d.word2vec', encoding='utf-8', binary=False)

'''
预训练中文词向量：
https://github.com/Kyubyong/wordvectors
'''
#wordvec = Word2Vec.load('zh_google/zh.bin')
#print(model.most_similar('电脑'))

#设置参数
embedding_dim = 300     # 词嵌入维度
maxlen = 250            # 序列最大长度
max_features = 50000    # 最大特征数（词汇表大小）
batch_size = 32         # 每批数据量大小
filters = 256           # 1维卷积核个数
kernel_size = 3         # 卷积核长度
hidden_dims = 250       # 隐藏层维度
nb_classes = 15         # 分类个数
epochs = 5              # 迭代次数

try:
    data = pickle.load(open('data', 'rb'))
    labels = pickle.load(open('labels', 'rb'))
    tokenizer = pickle.load(open('tokenizer', 'rb'))
    word_index = pickle.load(open('word_index', 'rb'))
    class_names = pickle.load(open('class_names', 'rb'))
except:    
    stopwords = [line.strip() for line in open('stopwords.txt').readlines()]
    # encoding='GB18030'
    print('读取并准备文本数据...')
    time0 = time()
    dirname = 'sogou'
    txts = []
    labels = []
    for dir_path, dir_name, files in tqdm(os.walk(dirname)):
        #类别名称
        if dir_name:
    #        if dir_name in ['career', 'cul']:
    #            continue
            class_names = dir_name
        #类别目录下的文本
        if not dir_name:
            for file in tqdm(files[:21000]):
                words = []
                txt_name = os.path.join(dir_path, file)
                try:
                    f = open(txt_name, encoding='GB18030') #, encoding='GB18030'
                    for line in f.readlines(): #'GB18030'
                        text = jieba.cut(line)
                        text2 = [i for i in text if i not in stopwords]
                        words += text2
                    if len(words) == 0:
                        continue
                    words = ' '.join(words)
                    txts.append(words)
                    labels.append(class_names.index(os.path.split(dir_path)[1]))
                    f.close()
                except UnicodeDecodeError as e:
                    continue
# txts = np.array(txts)
    labels = to_categorical(labels, nb_classes)
    ##label为0~9共10个类别，keras要求格式为binary class matrices,转化一下
    time1 = time()
    print('数据读取 分词耗时: ', time1-time0)

    tokenizer = Tokenizer(num_words=max_features, lower=False)
    # 标记化;这个类用来对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示
    tokenizer.fit_on_texts(txts)
    # 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。
    sequences = tokenizer.texts_to_sequences(txts)
    #将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
    del txts
    
    word_index = tokenizer.word_index
    #一个dict，保存所有word对应的编号id，从1开始
    print('Found %s unique tokens.' % len(word_index))
    print('Pad sequences...')
    # 序列填充
    data = pad_sequences(sequences, maxlen=maxlen)

    """
    使用pad_sequences函数，将每一条评论都填充（pad）到一个矩阵中。
    “填充”可以让输入的维度保持一致，将每个序列的指定地方填充为零，直到序列的最大长度
    也要通过to_categorical函数将标签转换成向量，将它们转换成二元向量，其中1代表正面，0代表负面。
    """
    del sequences
    time2 = time()
    print('向量化耗时: ', time2-time1)
    
    pickle.dump(data, open('data', 'wb'))
    pickle.dump(labels, open('labels', 'wb'))
    pickle.dump(word_index, open('word_index', 'wb'))
    pickle.dump(tokenizer, open('tokenizer', 'wb'))
    pickle.dump(class_names, open('class_names', 'wb'))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.2, random_state=5)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')



'''
X_train, X_test, Y_train, Y_test = np.load('f:/X_train.npy'), np.load('f:/X_test.npy'),\
                                np.load('f:/Y_train.npy'), np.load('f:/Y_test.npy')
'''
'''
#创建embedding_matrix
embedding_matrix = np.zeros((max_features + 1, embedding_dim))
for word, i in word_index.items():
    if i >= max_features:
        continue
    try:
        embedding_vector = wordvec[word]
    except KeyError:
        continue
    embedding_matrix[i] = embedding_vector
'''

# 创建模型
print('Build model...')
#model = Sequential() # 序贯模型是多个网络层的线性堆叠。
model = Sequential([
    Embedding(max_features+1,
                    embedding_dim,
                    input_length=maxlen),
    Dropout(0.2),
    Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1),
    GlobalMaxPooling1D(),
    Dense(hidden_dims),
    Dropout(0.2),
    Activation('relu'),
    Dense(nb_classes, activation='softmax')
])
# 高效的嵌入层，将词汇的索引值映射为 embedding_dims 维度的词向量嵌入维数
model.add(Embedding(max_features+1,
                    embedding_dim,
                    input_length=maxlen))
#                    weights=[embedding_matrix],
#                    trainable=False))

# 1D 卷积层，将学习 filters 个 kernel_size 大小的词组卷积核
#model.add(Dropout(0.2))
"""#dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
# 注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。"""

# 卷积神经网络池化层和卷积层作用
# 1.invariance(不变性)，这种不变性包括translation(平移)，rotation(旋转)，scale(尺度)
# 2.保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力
#model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))

# 最大池化最大子采样函数取区域内所有神经元的最大值（max-pooling）
#model.add(GlobalMaxPooling1D())

# 添加一个原始隐藏层
#model.add(Dense(hidden_dims))
#model.add(Dropout(0.2))
#model.add(Activation('relu'))
""" 
ReLU 的优点：
Krizhevsky. 发现使用 ReLU 得到的SGD的收敛速度会比 sigmoid/tanh 快很多。有人说这是因为
它是linear，而且 non-saturating(非饱和神经元,应该是在卷积神经网络中的局部连接,不是全连接层)
相比于 sigmoid/tanh，ReLU 只需要一个阈值就可以得到激活值，而不用去算一大堆复杂的运算。
ReLU 的缺点： 就是训练的时候很”脆弱”，很容易就”die”了. 什么意思呢？"""
# 增加一个全连接层，使用softmax获得分类
#model.add(Dense(nb_classes, activation='softmax'))#nb_classes添加输出节点
"""
得到概率最大的输出类似于向量的元素（单位化）标准化，使得在对output层每个节点概率求和值为1，
方便分类（classification）
"""

#model.summary()#打印出模型概况，它实际调用的是keras.utils.print_summary
"""
模型我们使用交叉熵损失函数，最优化方法选用adam
在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：
优化器optimizer：该参数可指定为已预定义的优化器名，如rmsprop、adagrad，或一个Optimizer类的对象，
损失函数loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，
如categorical_crossentropy、mse，也可以为一个损失函数。详情见losses
指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。
指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,
或一个完成metric_name - > metric_value映射的字典.请参考性能评估

"""

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# trainning
"""batch深度学习的优化算法，说白了就是梯度下降。每次的参数更新有两种方式。
第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。
这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，
不支持在线学习，这称为Batch gradient descent，批梯度下降。
另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，
stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，
hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
"""
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          # epochs指的就是训练过程中数据将被“轮”多少次，就这样。
          validation_data=(X_test, Y_test))

# 模型评估
score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('NLP_model' + str(datetime.now()))


#model predict
def predict(txt_dir):
    stopwords = [line.strip() for line in open('stopwords.txt').readlines()]
    print('读取数据...')
    txts = []
    for file in tqdm(os.listdir(txt_dir)):
        words = []
        txt_name = os.path.join(txt_dir, file)
        try:
            f = open(txt_name, encoding='utf-8') #, encoding='GB18030'
            for line in f.readlines(): #'GB18030'
                text = jieba.cut(line)
                text2 = [i for i in text if i not in stopwords]
                words += text2
            if len(words) == 0:
                continue
            words = ' '.join(words)
            txts.append(words)
            f.close()
        except:
            continue
    sequences = tokenizer.texts_to_sequences(txts)
    del txts
    data = pad_sequences(sequences, maxlen=maxlen)
    pred = model.predict_classes(data) # predict_classes,
    pred2 = [class_names[i] for i in pred]
    return pred, pred2
