#!/user/bin/python
# coding:utf-8
'''
1、文章分句
2、句子剔除停用词
3，句子中遍历关键词
4、计算句子得分；关键词之间的距离小于"门槛值"，（单词间的距离如果两个关键词之间有5个以上的其他词，
就可以把这两个关键词分在两个簇）它们就被认为处于同一个簇之中 对于每个簇，都计算它的重要性分值；例如：
其中的簇一共有7个词，其中4个是关键词。因此，它的重要性分值等于 ( 4 x 4 ) / 7 = 2.3
5、计算句子的平局值和方差，利用句子得分大于avg+0.5*std的值就返回句子作为摘要
'''
import nltk
import numpy
import jieba
import codecs
#分句
def sent_tokenizer(texts):
    global token
    start=0
    i=0#每个字符的位置
    sentences=[]
    punt_list='.!?。！？'
    for text in texts:
        if text in punt_list and token not in punt_list: #检查标点符号下一个字符是否还是标点
            sentences.append(texts[start:i+1])#当前标点符号位置
            start=i+1#start标记到下一句的开头
            i+=1
        else:
            i+=1#若不是标点符号，则字符位置继续前移
            token=list(texts[start:i+2]).pop()#取下一个字符
    if start<len(texts):
        sentences.append(texts[start:])#这是为了处理文本末尾没有标点符号的情况
    return sentences

#停用词
def load_stopwordslist(path):
    print('load stopwords...')
    stoplist=[line.strip() for line in codecs.open(path,'r',encoding='utf8').readlines()]
    stopwrods={}.fromkeys(stoplist)#创建一个新的字典，seq =stoplist- 这是用于字典键准备的值的列表
    return stopwrods
 #句子得分
def _score_sentences(sentences,topn_words):
    scores=[]
    sentence_idx=-1
    for s in [list(jieba.cut(s)) for s in sentences]:
        sentence_idx+=1
        word_idx=[]
        for w in topn_words:
            try:
                word_idx.append(s.index(w))#关键词出现在该句子中的索引位置
            except ValueError:#w不在句子中
                pass
        word_idx.sort()
        if len(word_idx)==0:
            continue
        #对于两个连续的单词，利用单词位置索引，通过距离阀值计算族
        #"簇"（cluster）表示关键词的聚集。所谓"簇"就是包含多个关键词的句子片段
        clusters=[]
        cluster=[word_idx[0]]
        i=1
        while i<len(word_idx):
            if word_idx[i]-word_idx[i-1]<5:#CLUSTER_THRESHOLD=5#单词间的距离如果两个关键词之间有5个以上的其他词，就可以把这两个关键词分在两个簇
                cluster.append(word_idx[i])
            else:
                clusters.append(cluster[:])
                cluster=[word_idx[i]]
            i+=1
        clusters.append(cluster)
        #对每个族打分，每个族类的最大分数是对句子的打分
        max_cluster_score=0
        for c in clusters:
            significant_words_in_cluster=len(c)
            total_words_in_cluster=c[-1]-c[0]+1
            # 关键词之间的距离小于"门槛值"，它们就被认为处于同一个簇之中
            # 对于每个簇，都计算它的重要性分值；例如：其中的簇一共有7个词，其中4个是关键词。因此，它的重要性分值等于 ( 4 x 4 ) / 7 = 2.3
            score=1.0*significant_words_in_cluster*significant_words_in_cluster/total_words_in_cluster

            if score>max_cluster_score:
                max_cluster_score=score
        scores.append((sentence_idx,max_cluster_score))
    return scores
#摘要
def summarize(text,number):

    stopwords=load_stopwordslist('stopwords.txt')#加载停用词
    sentences=sent_tokenizer(text)#分句子
    words=[w for sentence in sentences for w in jieba.cut(sentence) if w not in stopwords if len(w)>1 and w!='\t']
    #\t横向跳到下一制表符位置 簇的长度
    wordfre=nltk.FreqDist(words) #簇频率分布情况
    topn_words=[w[0] for w in sorted(wordfre.items(),key=lambda d:d[1],reverse=True)][:100]#单词的数量；key=lambda d:d[1]按照value进行排序
    scored_sentences=_score_sentences(sentences,topn_words)#计算文章中的句子得分
    #approach 1,利用均值和标准差过滤非重要句子
    avg=numpy.mean([s[1] for s in scored_sentences])#文章中所有句子得分的均值
    std=numpy.std([s[1] for s in scored_sentences])#文章中所有句子得分的标准差
    mean_scored=[(sent_idx,score) for (sent_idx,score) in scored_sentences if score>(avg+0.5*std)]#如果句子得分大于avg+0.5*std的值就返回句子
    #approach 2，返回top n句子直接返回句子得分高的句子
    top_n_scored=sorted(scored_sentences,key=lambda s:s[1])[-number:]
    top_n_scored=sorted(top_n_scored,key=lambda s:s[0])
    return dict(top_n_summary=[sentences[idx] for (idx,score) in top_n_scored],mean_scored_summary=[sentences[idx] for (idx,score) in mean_scored])
if __name__=='__main__':
    f=open('new.txt').read()
    dict=summarize(f,3)
    print('-----------approach 1-------------')
    for sent in dict['top_n_summary']:
        print(sent)
    print('-----------approach 2-------------')
    for sent in dict['mean_scored_summary']:
        print(sent)