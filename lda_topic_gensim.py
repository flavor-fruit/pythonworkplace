# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging
import  numpy as np
import lda
import lda.datasets
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from gensim import corpora, models, similarities

def get_topic_term(lda, topicid,word_num):
    """
    Return a list of `(word_id, probability)` 2-tuples for the most
    probable words in topic `topicid`.

    Only return 2-tuples for the topn most probable words (ignore the rest).

    """
    topic = lda.state.get_lambda()[topicid]
    topic = topic / topic.sum()  # normalize to probability distribution

    return [topic[id] for id in range(0, word_num)]

def dealWithLda(documents,lda,path):
    dictionary = corpora.Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    X= corpus2matrix(lda[corpus])
    data = pd.DataFrame(X)
    data.to_csv(path)



def creatLDA(documents,topic_num):
    dictionary = corpora.Dictionary(documents)
    # print dictionary
    # print dictionary.token2id
    corpus = [dictionary.doc2bow(text) for text in documents]
    # print "corpus"
    # print corpus
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #     print doc
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=topic_num, alpha=5, eta=0.1,iterations=500)

    topic_word = []
    file = open('topic.csv','w')
    for i in range(0, topic_num):
       # print  get_topic_term(lda, i, len(dictionary))
        words = topic2words(lda.show_topic(i, len(dictionary)))
        file.write(','.join(words)+'\n')
        topic_word.append(words)
    file.close()
    # narray = np.array(topic_word)
    # print narray.shape
    # data = pd.DataFrame(narray)
    # data.to_csv('topic.csv')
    return lda

def topic2words(topic):
    words=[]
    for word in topic:

            words.append(word[0].replace('\n','!!'))


    return words

def corpus2matrix(corpus):
    data = []
    rows = []
    cols = []
    line_count = 0
    for line in corpus:
        for elem in line:
            rows.append(line_count)
            cols.append(elem[0])
            data.append(elem[1])
        line_count += 1
    sparse_matrix = csr_matrix((data, (rows, cols)))  # 稀疏向量
    matrix = sparse_matrix.toarray()  # 密集向量
    return matrix
#def dealWithL(real_documents,topic_num):

def get_topic_term(lda, topicid,word_num):
    """
    Return a list of `(word_id, probability)` 2-tuples for the most
    probable words in topic `topicid`.

    Only return 2-tuples for the topn most probable words (ignore the rest).

    """
    topic = lda.state.get_lambda()[topicid]
    topic = topic / topic.sum()  # normalize to probability distribution

    return [topic[id] for id in range(0, word_num)]


if __name__ == '__main__':
    train_documents = []
    for i in range(1, 9):
        for j in range(1, 1001):
            word_in_file = {}
            f = open('E:/paper/split_result/train/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8').split(',')
            f.close()
            train_documents.append(txtlist)
    lda = creatLDA(train_documents,10)

    dealWithLda(train_documents,lda,'train_topic.csv')

    test_documents = []
    for i in range(1, 9):
        for j in range(1001, 1251):
            word_in_file = {}
            f = open('E:/paper/split_result/test/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8').split(',')
            f.close()
            test_documents.append(txtlist)
    dealWithLda(test_documents,lda,'test_topic.csv')
