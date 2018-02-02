# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 01:13:53 2017

@author: Atlantis
"""



from __future__ import print_function
import os
import re
import string
import jieba
import numpy as np
from time import time
from tqdm import tqdm
from gensim import corpora, models, similarities
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split
from sklearn import metrics


#word2vect

def fenci(txt):
    empty = '将 于 一 地 的 和 及 也 之 在 只  仅 与 了 即 也 若 比 及 我 为 他 是 他们 就 都 以 到 她 '.split()
    symbol = string.punctuation +'－-[]％（） ～·，：、。；“”【】±–—①②' #ASCII 标点符号，空格和数字
    try:
        for line in open(txt, encoding='mbcs').readlines():  #
            line = re.sub('\d+(\.)?\d*','',line)    #去掉数字
            line = re.sub('[a-zA-Z]+','',line)     #去掉字母
            line = re.sub('[\\s]*','',line)       #去掉换行符
            text = list(jieba.cut(line))
            text2 = [i for i in text if i not in empty+list(symbol)]
            return text2
    except UnicodeDecodeError as e:
        print (e)
        return None


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2
nb_classes = 15

vocab = corpora.Dictionary()
vocab.load('词典')

print('读取并准备文本数据...')
time0 = time()
txts = []
labels = []
sogou = '/home/dl/NLP/sogou'
for dir_path, dir_name, files in os.walk(sogou):
	#类别名称
	if dir_name:
		class_names = dir_name
	#类别目录下的图片
	if not dir_name:
		for file in files:
			txt_name = os.path.join(dir_path, file)
			txts.append(txt_name)
			labels.append(class_names.index(os.path.split(dir_path)[1]))
txts = np.array(txts)
labels = to_categorical(labels, nb_classes)
X_train, X_test, Y_train, Y_test = train_test_split(txts, labels, test_size=0.2, random_state=5)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


#将单个文本转换为词向量格式
def txt_trans(X):
    Xs = []
    if isinstance(X, str):
        X = [X]
    for txt_name in X:
        text = fenci(txt_name)
        text_vec = []
        for x in text:
            index = vocab.get(x)
            if index == None:
                index = 0
            text_vec.append(index)
        Xs.append(text_vec)
    Xs = sequence.pad_sequences(Xs, maxlen=maxlen)
    Xs = np.array(Xs)
    return Xs

    #Pad sequences (samples x time)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(X_test, Y_test))

Y_train_pred = []
for X in tqdm(X_train[:500]):
    y = model.predict_classes(txt_trans([X]))
    Y_train_pred.append(y)
Y_pred = []
for X in tqdm(X_test[:500]):
    y = model.predict_classes(txt_trans([X]))
    Y_pred.append(y)
print('训练预测准确度: ', metrics.accuracy_score(Y_train_pred, Y_train[:500].argmax(1)))
print('测试预测准确度: ', metrics.accuracy_score(Y_pred, Y_test[:500].argmax(1)))

model.save('NLP0917.h5')

score, acc = model.evaluate(X_test, Y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)



