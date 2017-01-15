# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import numpy as np






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




data=pd.read_table('E:/eclipse_workplace/JGibbLDA-v.1.0/models/sougou/topic-150/model-final.phi',sep=' ',header=None)
a=np.array(data)
# print a.shape
# print a
res_cov=np.std(a,axis=0)
print res_cov
print type(res_cov)

cov_word = {}

for i in range(0, len(res_cov)):
    cov_word[i] =res_cov[i]
cov_word_sorted = sorted(cov_word.iteritems(), key=lambda asd: asd[1], reverse=True)
print cov_word_sorted

dictionary_id={}
f=open('E:\\paper\\dic.txt','r')
txtlist = f.read().strip('\n').replace(',',' ').decode('utf-8').split('\n')
for i in range(len(txtlist)):
    dictionary_id[i]=txtlist[i]


word_sorted_index = []
for i in range(0, len(cov_word_sorted)):
    word_sorted_index.append(cov_word_sorted[i])
print word_sorted_index

f = open("E:/paper/sougou/lda_desend/all.txt", 'a')
for i in word_sorted_index:
    for key, value in dictionary_id.iteritems():
        if value == i[0]:
             print key
             f.write(key.encode('utf-8') + ":" + str(i[1]))
             f.write('\n')
f.close()
