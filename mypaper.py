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





if __name__ == '__main__':
    # 给原始文档分词
    dictionary=[]
    real_documents=[]
    for i in range(1,9):
        for j in range(1,100):
           onewordbag=splitSentence('E:/paper/data/'+str(i)+'/'+str(j)+'.txt')
           s=" ".join(onewordbag)
           real_documents.append(s)
           for k in onewordbag:
               dictionary.append(k)
    #计算tfidf
    # vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    # transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(real_documents))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    # word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    # tfidf_matrix = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    # for i in range(len(tfidf_matrix)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    #     print u"-------这里输出第", i, u"类文本的词语tf-idf权重------"
    #     for j in range(len(word)):
    #          print word[j], tfidf_matrix[i][j]
    #计算词频
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(real_documents)#fit_transform将文本转为词频矩阵
    word = vectorizer.get_feature_names()#获取词袋模型中的所有词语
    analyze = vectorizer.build_analyzer()
    tf_matrix=X.toarray()
    #打印特征向量文本内容
    print 'Feature length ：'+str(len(word))
    # for j in range(len(word)):
    #     print word[j]
    #打印每类文本词频矩阵
    # print 'TF weight:'
    # for i in range(len(tf_matrix)):
    #     for j in range(len(word)):
    #         print tf_matrix[i][j]

    #LDA模块
    #指定主题数，迭代次数
    m=100
    model=lda.LDA(n_topics=m,n_iter=1000,random_state=1)
    #model.fit(tfidf_matrix)
    model.fit(tf_matrix)
    topic_word=model.topic_word_ #topic_word中一行对应一个topic，一行之和为1
    #获取每个topic下权重最高的n个单词
    n=50
    for i,topic_dist in enumerate(topic_word):
        topic_words=np.array(word)[np.argsort(topic_dist)][:n]
        #print ('*Topic {}\n-{}'.format(i ,' '.join(topic_words)))
    pri_dic=topic_words.tolist()
    vsm_dict=[]
    for i in range(0,len(pri_dic)) :
        for j in range(0,len(pri_dic[0])):
            vsm_dict.append(topic_words[i][j])



     #去除重复词语
    vsm_dict=list(set(vsm_dict))
    print len(vsm_dict)
    for i in range(len(vsm_dict)):
        print vsm_dict[i].encode("gbk")
    # 计算tfidf
    tfidf_vectorizer = TfidfVectorizer(vocabulary=vsm_dict)
    real_vec = tfidf_vectorizer.fit_transform(real_documents)
    real_vec=real_vec.toarray()
    print real_vec.sum(axis=0)

