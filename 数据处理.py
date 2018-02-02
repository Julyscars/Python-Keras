# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:35:57 2017

@author: byq
"""

import os
import numpy as np
import jieba
import pandas as pd


path = r'D:\NLP\NER\BosonNLP_NER_6C\BosonNLP_NER_6C.txt'
text = open(path, encoding='utf-8').read()
text = text.split('\n')
output = open('output.txt', 'w+', encoding='utf-8')

entity = ['time', 'location', 'person_name', 'org_name', 'company_name',
          'product_name']
#stopwords = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()]

for para in text:
    para = para.replace('{{', '\n')
    para = para.replace('}}', '\n')
    para = para.replace(':', ' ')
    para = para.split('\n')
    for i in para:
        if i.strip() == '':
            continue
        line = i.split()
        if line[0].strip() in entity:
            try:
                output.writelines([line[1], ' ', line[0], '\n'])
            except:
                continue
        else:
            line = list(jieba.cut(line[0]))
            for word in line:
    #            if word in stopwords:
    #                continue
                output.writelines([word, ' ', 'O', '\n'])
    output.writelines('\n')
output.close()



