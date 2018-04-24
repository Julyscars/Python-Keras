# -*- coding: utf-8 -*-

import logging
from gensim.models import word2vec

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.LineSentence("data1.txt")
    model = word2vec.Word2Vec(sentences, size=250,window=5,min_count=5,negative=3, sample=0.001, hs=1, workers=4)
    #保存模型，供日後使用
    model.save("word2vec.model")


if __name__ == "__main__":
    main()
