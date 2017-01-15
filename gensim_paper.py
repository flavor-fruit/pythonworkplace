# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging
import  numpy as np
import lda
import lda.datasets

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

#分词
import re
import sys
sys.path.append('../')
import jieba
import jieba.posseg as pseg
def splitSentence(filepath):
    stopwords = {}.fromkeys([line.rstrip() for line in open('E:/paper/stopwords.txt')])
    f=open(filepath,'r')
    txtlist = f.read().strip().decode('utf-8')
    xx = u"([\u4e00-\u9fa5]+)"
    pattern = re.compile(xx)
    results = pattern.findall(txtlist)
    str = ''.join(results)
    words = pseg.cut(str)
    f.close()
    usefulwords=[]
    for w in words:
        if( (w.flag=='n'or w.flag=='v')and len(w.word)>1):
                if w.word.encode('gbk') not in stopwords:
                    usefulwords.append(w.word)
                    #print w.word
    return usefulwords


def get_topic_term(lda, topicid,word_num):
    """
    Return a list of `(word_id, probability)` 2-tuples for the most
    probable words in topic `topicid`.

    Only return 2-tuples for the topn most probable words (ignore the rest).

    """
    topic = lda.state.get_lambda()[topicid]
    topic = topic / topic.sum()  # normalize to probability distribution

    return [topic[id] for id in range(0, word_num)]

def dealWithLda(real_documents,topic_num):
    from gensim import corpora, models, similarities
    dictionary = corpora.Dictionary(real_documents)
    # print dictionary
   # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in real_documents]
    # print "corpus"
    # print corpus
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #     print doc
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num)
    corpus_lda = lda[corpus_tfidf]
    vsm_dic = []
    #print get_topic_term(lda, 0, 100)
    topic_word = []
    for i in range(0, topic_num):
       # print  get_topic_term(lda, i, len(dictionary))
        topic_word.append(get_topic_term(lda, i, len(dictionary)))

    #print topic_word
    #topic_word_matrxi = np.matrix(topic_word)
   # print topic_word_matrxi
    return topic_word,dictionary.token2id



#def dealWithL(real_documents,topic_num):








#计算主题-词矩阵的标准差 输出标准差较大的词语的索引
def cov_compute(a):
    import numpy as np
    import math
    import itertools
    b=np.matrix(a)
    print b
    num_row=b.shape[0]#行数
    num_col=b.shape[1]
    print  num_row

    cov_word={}
    sum_col= b.sum(axis=0).tolist()#矩阵每列相加 得到一个一维的矩阵
    sum_col=list(itertools.chain.from_iterable(sum_col))
    print sum_col

    for i in range(0,len(sum_col)):
        qud = 0
        avg_col=sum_col[i]/num_row*1.0
        for j in range(0,num_row):
            qud=qud+(a[j][i]-avg_col)*(a[j][i]-avg_col)
        cov_word[i]=math.sqrt((qud/num_row)*1.0)
    #print cov_word
    cov_word_sorted=sorted(cov_word.iteritems(),key=lambda asd:asd[1],reverse=True)
   # print cov_word_sorted
    word_sorted_index=[]
    for i in range(0,len(cov_word_sorted)):
        word_sorted_index.append(cov_word_sorted[i])
    print word_sorted_index
    return word_sorted_index



if __name__ == '__main__':
    real_documents=[]
    for i in range(1,9):
        for j in range(10,1010):
            f = open('E:/paper/sougou/split_result/train/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8').split(',')

            f.close()
            real_documents.append(txtlist)
    topic_word,dictionary_id=dealWithLda(real_documents,30)
    # print "the length of dictionary is :"
    # print len(dictionary)
    # print dictionaryi
    print dictionary_id
    word_sorted_index=cov_compute(topic_word)
    f = open("E:/paper/sougou/lda_desend/all1.txt", 'a')
    for i in word_sorted_index:
        for key, value in dictionary_id.iteritems():
            if value ==i[0]:
                #print key
                f.write(key.encode('utf-8')+":"+str(i[1]))
                f.write('\n')
    f.close()




