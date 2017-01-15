# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import re
topic_number = [10,20,200]

for i in topic_number:
    f = open('E:\\eclipse_workplace\\JGibbLDA-v.1.0\models\\topic-%d\\model-final.twords'%i,'r')
    content = f.read()
    p1 = r'\bTopic.*\b:\n'
    pattern1 = re.compile(p1)

    result = pattern1.sub('',content).replace('\t','').strip('\n')
    p2 = r' \b.*\b'
    pattern2 = re.compile(p2)
    all_word = pattern2.sub('',result)
    word_list = list(set(all_word.split('\n')))
    word_dict = '\n'.join(word_list)
    f2 = open('E:\\paper\\lda_word\\%d\\word_dict.txt'%i,'w')
    f2.write(word_dict)
