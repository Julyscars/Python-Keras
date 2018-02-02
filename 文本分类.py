# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 21:57:34 2017

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
from gensim.models import Word2Vec
from gensim import corpora, models, similarities
from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
#import gensim.models.word2vec.BrownCorpus
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding
from keras.layers import LSTM
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


max_features = 20000
embedding_dims = 60
filters = 250
kernel_size = 3
hidden_dims = 250
maxlen = 400  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
nb_classes = 15

'''
#生成词向量
class NewsCorpus():
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""
    def __init__(self):
        self.dirname = r'C:\MyPy\数据\sogou\Reduced'
    def __iter__(self):
        for dire in tqdm(os.listdir(self.dirname)):
            for file in tqdm(os.listdir(os.path.join(self.dirname, dire))):
                txt = os.path.join(self.dirname, dire, file)
                text = fenci(txt)
                if not text:  # don't bother sending out empty sentences
                    continue
                yield text
news = NewsCorpus()
wv = Word2Vec(news, size=embedding_dims, window=5, min_count=5, workers=4)
wv.save('词向量')
'''

wv = Word2Vec.load('词向量')
#获得权重
weights = wv.wv.syn0
#获得词库
vocab = dict([(k, v.index) for k,v in wv.wv.vocab.items()])

print('读取并准备文本数据...')
time0 = time()
txts = []
labels = []
sogou = r'C:\MyPy\数据\sogou\Reduced'
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
model.add(Embedding(input_dim=weights.shape[0],
                    output_dim=weights.shape[1],
                    weights=[weights],
                    input_length = maxlen,
                    trainable=False)
            )
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(15, activation='sigmoid'))

# try using different optimizers and different optimizer configs
#print('模型开始编译...')
time1 = time()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

time2 = time()
print('模型编译完毕，耗时：%d秒。\n模型开始训练...' % (time2-time1))
nb_epoch = X_train.shape[0] // batch_size
for i in tqdm(range(nb_epoch)):
    X = txt_trans(X_train[i*batch_size:(i+1)*batch_size])
    Y = Y_train[i*batch_size:(i+1)*batch_size]
    loss = model.train_on_batch(X, Y)
    print('loss: ',loss)
print('模型训练完毕，耗时：%d秒。\n模型开始运行预测程序...' % (time()-time2))

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


'''
#CNN
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
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
'''
