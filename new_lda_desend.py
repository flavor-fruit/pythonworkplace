# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd
import numpy as np






data=pd.read_table('E:/eclipse_workplace/JGibbLDA-v.1.0/models/topic-50/model-final.phi',sep=' ',header=None)
a=np.array(data.dropna(axis=1))
print a.shape
print a
cov_word = np.std(a,axis=0)

dic_frame = pd.read_table('E:/eclipse_workplace/JGibbLDA-v.1.0/models/topic-50/wordmap.txt',sep=' ')
cov_frame = pd.DataFrame(cov_word,index=dic_frame.index,columns=None)

sort_cov = cov_frame.sort_values(by=0,ascending=False)
sort_cov.to_csv('topic_word_lda.csv',columns=None)
