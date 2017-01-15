#-*- encoding:utf-8 -*-
import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import sys
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf8')
#计算df值 通过df值降维
docs = []

for i in range(1,9):
    for j in range(1,1001):
        ciyuji=[]
        f2= open('E:/paper/split_result/train/'+str(i)+'/'+str(j)+'.txt', 'r')
        text1 = f2.read().strip().replace(',',' ').decode('utf-8')
        docs.append(text1)
        f2.close()
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(docs)
        #if word in ciyuji:

df = text._document_frequency(X)

data = pd.DataFrame(df,index=vectorizer.get_feature_names())
data.to_csv('E:/paper/df/origin.csv',header=False)

        #print word.encode('utf-8')+":"+str(count)


