#!/usr/bin/python  
#-*- encoding:utf-8 -*-
import re
import sys
sys.path.append('../')
import jieba       #导入jieba模块
import jieba.posseg as pseg

#分词
def splitSentence(inputFile, outputFile):
    stopwords = {}.fromkeys([ line.rstrip() for line in open('E:/paper/stopwords.txt') ])
    f= open(inputFile, 'r')
    txtlist = f.read().strip().decode('utf-8').replace('\n',' ')
    xx = u"([\u4e00-\u9fa5]+)"
    pattern = re.compile(xx)
    results  = pattern.findall(txtlist)
    str = ''.join(results)
    words =pseg.cut(str)
    f.close()
    count=0
    for w in words:
        count=count+1
        if((w.flag == 'n' or w.flag == 'v') and len(w.word) > 1):
                if w.word.encode('gbk') not in stopwords:
                    f=open(outputFile,'a')
                    f.write(w.word.encode('utf-8'))
                    f.write(',')

    f.close()




#去除重复词语
def extract(inputFile, outputFile):
    f= open(inputFile, 'r')
    txtlines = f.read().strip().decode('utf-8')
    f.close()
    txtlines = txtlines.replace('\n','').split(u',')
    wordset=list(set(txtlines))
    #wordset=set()
    # for line in txtlines:
    #     wordset.add(line)
    # wordset = list(wordset)
    count=0
    f=open(outputFile,'a')
    for word in wordset:
        count=count+1
        f.write(word.encode('utf-8'))
        f.write(',')

    f.close()

if __name__ == '__main__':
    for i in range(1,9):
        for j in range(1,1001):
           splitSentence('E:/paper/data/'+str(i)+'/'+str(j)+'.txt', 'E:/paper/split_result/train/'+str(i)+'/'+str(j)+'.txt')
           splitSentence('E:/paper/data/'+str(i)+'/'+str(j)+'.txt', 'E:/paper/split_result/train/all.txt')
    extract( 'E:/paper/split_result/train/all.txt', 'E:/paper/split_result/train/all1.txt')
#对测试集分词处理
    for i in range(1,9):
         for j in range(1001,1501):
              splitSentence('E:/paper/data/'+str(i)+'/'+str(j)+'.txt', 'E:/paper/split_result/test/'+str(i)+'/'+str(j)+'.txt')
