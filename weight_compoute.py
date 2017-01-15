# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')


import pandas as pd
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer




def tfidf_weight(real_documents,dicOfUser,outputpath):
    #

    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=len(dicOfUser),vocabulary=dicOfUser)
    tfidf = tfidf_vectorizer.fit_transform(real_documents)


    data = pd.DataFrame(tfidf.toarray(),columns=tfidf_vectorizer.get_feature_names())
    #data = add_label_train(num,data)
    data.to_csv(outputpath)



def add_label_train(num,matrix):
    label = [i / 1000 + 1 for i in range(num)]
    matrix['Col_sum'] = label
    return matrix

def add_label_test(num,matrix):
    label = [i / 250 + 1 for i in range(num)]
    matrix['Col_sum'] = label
    return matrix

number_list =[10,20,200]  #lda topic number
#number_lsit=[]   #mi number

for number in number_list:
    train_documents = []
    dfile = open('E:\\paper\\lda_word\\%d\\word_dict.txt'%number, 'r')
    dictionary = dfile.read().decode('utf-8').strip('\n').split('\n')
    for i in range(1, 9):
        for j in range(1, 1001):
            word_in_file = {}
            f = open('E:/paper/split_result/train/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().replace(',', ' ').decode('utf-8')
            f.close()
            train_documents.append(txtlist)

    # 为降维的词典
    # dictionary= dfile.read().strip().replace(',',' ').strip(' ').decode('utf-8').split(' ')
    tfidf_weight(train_documents, dictionary, 'E:\\paper\\lda_word\\%d\\train_tfidf.csv' % number)

    test_documents = []
    for i in range(1, 9):
        for j in range(1001, 1251):
            word_in_file = {}
            f = open('E:/paper/split_result/test/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().replace(',', ' ').decode('utf-8')
            f.close()
            test_documents.append(txtlist)

    # dictionary= dfile.read().strip().replace(',',' ').strip(' ').decode('utf-8').split(' ')
    tfidf_weight(test_documents, dictionary, 'E:\\paper\\lda_word\\%d\\test_tfidf.csv' % number)