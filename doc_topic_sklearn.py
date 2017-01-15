# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import pandas as pd
from sklearn.decomposition import  LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def add_label(num,matrix):
    label = [i / 1000 + 1 for i in range(num)]
    matrix['Col_sum'] = label
    return matrix



train_documents = []
for i in range(1,9):
    for j in range(1,1001):
         word_in_file={}
         f= open( 'E:/paper/split_result/train/'+str(i)+'/'+str(j)+'.txt', 'r')
         txtlist = f.read().strip().replace(',',' ').decode('utf-8')
         f.close()
         train_documents.append(txtlist)

lda = LatentDirichletAllocation(n_topics=100,max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=-0)
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(train_documents)
lda.fit(X)#训练过程
data = pd.DataFrame(lda.transform(X))
data.to_csv('E:\\paper\\lda_topic\\test.csv')

test_documents = []
for i in range(1,9):
    for j in range(1001,1251):
         word_in_file={}
         f= open( 'E:/paper/split_result/test/'+str(i)+'/'+str(j)+'.txt', 'r')
         txtlist = f.read().strip().replace(',',' ').decode('utf-8')
         f.close()
         test_documents.append(txtlist)


Y = vectorizer.fit_transform(test_documents)

data = pd.DataFrame(lda.transform(Y))
data.to_csv('E:\\paper\\lda_topic\\test.csv')