# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:19:37 2017

@author: byq
"""
import math
import json
import os
import numpy as np
import jieba
import pandas as pd
from collections import Counter
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM,Activation
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


min_freq = 2
EPOCHS = 50
EMBED_DIM = 200
BiRNN_UNITS = 200

# Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score
def classification_report(y_true, y_pred, labels):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total', sum(report2[0]) / N, sum(report2[1]) / N, sum(report2[2]) / N, N) + '\n')
# 数据读取
data = open('data/output.txt', encoding='utf-8').read()
#data = data.replace("\n\n\n\n","\n\n").replace("\n\n\n","\n\n").replace("\n\n\n\n\n","\n\n")
data = [[row.split() for row in para.split('\n')] for para in data.strip().split('\n\n\n')]
"""
for para in data.strip().split('\n\n\n')每篇文章进行切分;
for row in para.split('\n')每行文本切分;
row.split()每行分词
"""



test_size = int(len(data) * 0.1)
train, test = data[:-test_size], data[-test_size:]

word_counts = Counter(row[0] for sample in train for row in sample)
vocab = ['<pad>', '<unk>'] + [w for w, f in word_counts.items() if f >= min_freq]
# in alphabetic order
chunk_tags = sorted(list(set(row[1] for para in train + test for row in para)))
class_labels = chunk_tags

def process_data(data, vocab, chunk_tags, maxlen=None):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    word2idx = dict((w, i) for i, w in enumerate(vocab))
    x = [[word2idx.get(w[0], 1) for w in s] for s in data]  # set to <unk> (index 1) if not in vocab
    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    x = pad_sequences(x, maxlen)  # left padding
    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
    y_chunk = np.expand_dims(y_chunk, 2)
    return x, y_chunk

train_X, train_Y = process_data(train, vocab, chunk_tags)
test_X, test_Y = process_data(test, vocab, chunk_tags)
# --------------
# 1. Regular CRF
# --------------
'''
print('==== training CRF ====')

model = Sequential()
model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()

model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.fit(train_X, train_Y, epochs=EPOCHS, validation_data=[test_X, test_Y])

test_y_pred = model.predict(test_X).argmax(-1)[test_X > 0]
test_y_true = test_Y[test_X > 0]

print('\n---- Result of CRF ----\n')
classification_report(test_y_true, test_y_pred, class_labels)
'''


print('==== training BiLSTM-CRF ====')

# model = Sequential()
# model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
# model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
# model.add(Activation('tanh'))
model = Sequential([
    Embedding(len(vocab),EMBED_DIM,mask_zero=True),
    Bidirectional(LSTM(BiRNN_UNITS // 2,return_sequences=True)),
    Activation('tanh')
])
def step_decay(epoch):
    initial_lrate = 0.000001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    lrate=initial_lrate
    return lrate

lrate = LearningRateScheduler(step_decay)

crf = CRF(len(class_labels), sparse_target=True)
model.add(crf)
model.summary()

model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.fit(train_X, train_Y, epochs=EPOCHS, validation_data=[test_X, test_Y],callbacks=None,batch_size=200)


test_y_pred = model.predict(test_X).argmax(-1)[test_X > 0]
test_y_true = test_Y[test_X > 0]

param_out=open("/home/dl/NLP/NER/model/zh_param","w")
out_json={"class_labels":class_labels,"class_labels_len":len(class_labels),"vocab_len":len(vocab),"EMBED_DIM":EMBED_DIM,"BiRNN_UNITS":BiRNN_UNITS,"vocab":vocab}
param_out.write(json.dumps(out_json))
param_out.flush()
param_out.close()
save_load_utils.save_all_weights(model,"/home/dl/NLP/NER/model/zh_ner_model")

print('\n---- Result of BiLSTM-CRF ----\n')

classification_report(test_y_true, test_y_pred, class_labels)