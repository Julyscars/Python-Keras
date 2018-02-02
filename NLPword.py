#coding:utf-8
import gensim
import jieba
from gensim import corpora
documents = [
    '拍照反光一直是摄影爱好者较为苦恼的问题',
    '尤其是手机这种快速拍照设备的成像效果总是难以令人满意',
    '特别是抓拍的珍贵照片',
    '遇上反光照片基本作废',
    '而索尼最近研发的集成偏振片传感器',
    '似乎可以有效的解决拍照反光的问题'
]
texts = [jieba.lcut(document) for document in documents]
#构造字典 并 保存和加载
dictionary = corpora.Dictionary(texts)
dictionary.save('mydict.dic')
print ('Tokens:Id')
print (dictionary.token2id)
new_dictionary = corpora.Dictionary.load('mydict.dic')
print(new_dictionary)

#构造新的文本并且获得他的向量
new_document = "索尼可以有效解决拍照的问题，佳能就不可以"
new_vector = dictionary.doc2bow(jieba.lcut(new_document))
print ('the vector of "%s": （tokenid,frequency)' % new_document)
print (new_vector)

#生成语料库
corpus = [ dictionary.doc2bow(text) for text in texts]
#序列化
corpora.MmCorpus.serialize('corpus.mm', corpus)
#重新加载预料
new_corpus = corpora.MmCorpus('corpus.mm')
print(len(new_corpus))
