# -*- coding: utf-8 -*-

import jieba
import logging
from gensim.models import word2vec
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # jieba custom setting.
    jieba.set_dictionary('jieba_dict/dict.txt.big')
    # load stopwords set
    stopword_set = set()
    with open('jieba_dict/stopwords.txt','r', encoding='utf-8') as stopwords:
        for stopword in stopwords:
            stopword_set.add(stopword.strip('\n'))
    output = open('data1.txt', 'w', encoding='utf-8')
    with open('data1.txt', 'r',encoding='utf-8') as content :
        for texts_num, line in enumerate(content):
            line = line.strip('\n')
            words = jieba.cut(line, cut_all=False)
            for word in words:
                if word not in stopword_set:
                    output.write(word + ' ')
            output.write('\n')

            if (texts_num + 1) % 10000 == 0:
                logging.info("已完成前 %d 行的斷詞" % (texts_num + 1))
    output.close()
    sentences = word2vec.LineSentence("data.txt")
    model = word2vec.Word2Vec(sentences, size=250,window=5,min_count=5,negative=3, sample=0.001, hs=1, workers=4)

    # 保存模型
    model.save("word2vec.model")
if __name__ == '__main__':
    main()

"""
参数解释：
1.sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。

2.size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。

3.window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。

4.min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。

5.negative和sample可根据训练结果进行微调，sample表示更高频率的词被随机下采样到所设置的阈值，默认值为1e-3。

6.hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。

7.workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。

详细参数说明可查看word2vec源代码。

2、训练后的模型保存与加载

[python] view plain copy
model.save(fname)
model = Word2Vec.load(fname)
3、模型使用（词语相似度计算等）


[python] view plain copy
model.most_similar(positive=['woman', 'king'], negative=['man'])
#输出[('queen', 0.50882536), ...]

model.doesnt_match("breakfast cereal dinner lunch".split())
#输出'cereal'

model.similarity('woman', 'man')
#输出0.73723527

model['computer']  # raw numpy vector of a word
#输出array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
1.使用`jieba` 去除停用断詞
```
python3 fenci.py
```
2.使用`gensim` 的 word2vec 模型训练
```
python3 train.py
```
3.测试训练的模型
```
python3 demo.py
"""
