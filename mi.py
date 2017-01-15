# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import math
#通过互信息降维
f= open('E:/paper/df/origin.txt', 'r')
word={}

while True:
    line =f.readline().decode('utf-8')
    if len(line)>1:

       line = line.split('\t')
       print line[0]
       print line[1]
       # for i in line:
       #   print i.encode('utf-8')
       #count=count+1
       #print count
       #print line[0].encode('utf-8')+':'+line[1].encode('utf-8')
       word[(line[0])]=line[1]
    else:
        break

temp=0

f1=open('E:/paper/sougou//mi/origin.txt','a')
for key,value in word.items():
    if int(value) == 0:
        print value
        value = int(value) + 1
    #print value
    for i in range(1,9):
        count=0
        for j in range(10,1010):
           f= open( 'E:/paper/sougou/split_result/train/'+str(i)+'/'+str(j)+'.txt', 'r')
           txtlist = f.read().strip().decode('utf-8')
           txtlist=txtlist.split(u',')
           if(txtlist.__contains__(key)):
              count=count+1
        if count>temp:
            temp=count

    mi=math.log10(temp*8.0/float(value))
    f1.write(key.encode('utf-8')+":"+str(mi))
    f1.write('\n')
f1.close()






