# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 18:20:56 2017

@author: byq
"""

import os
import numpy as np
from tqdm import tqdm
import jieba
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model


tokenizer = pickle.load(open('tokenizer', 'rb')) #反序列化对象,将对象的状态信息转换为可以存储或传输的形式的过程
word_index = pickle.load(open('word_index', 'rb'))
class_names = pickle.load(open('class_names', 'rb'))

#载入模型
print('Load Model...')
model = load_model('NLP_model')


#设置参数
maxlen = 250

#model predict
def predict(txt_dir):
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()] # line.strip() del space
    print('读取数据...')
    files = []
    txts = []
    for file in tqdm(os.listdir(txt_dir)):  #用于返回指定的文件夹包含的文件或文件夹的名字的列表。
                                            #这个列表以字母顺序。 它不包括 '.' 和'..' 即使它在文件夹中。
        files.append(file)
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
    pred = model.predict_classes(data) #predict_classes,
    pred2 = [class_names[i] for i in pred]
    print(['文件: %s, 预测类别: %s' % (i, j) for i,j in zip(files, pred2)])

if __name__ == '__main__':
    txt_dir = 'testset'
    predict(txt_dir)
