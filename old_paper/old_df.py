#-*- encoding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
#计算df值 通过df值降维
f= open('E:/paper/split_result/train/all1.txt', 'r')
#f=open('E:/paper/lda/3503/3503.txt','r')
#f=open('E:/paper/lda/5493/5493.txt','r')
#f=open('E:/paper/lda/8108/8108.txt','r')
#txtlines = f.read().strip().decode('utf-8')
txtlines = f.read().strip().decode('utf-8')
f.close()
#txtlines = txtlines.replace('\n','').split(u',')
txtlines = txtlines.split(u',')

for word in txtlines:
    print word
    if(len(word)!=0):
        count=0
        for i in range(1,9):
           for j in range(1,1001):
               ciyuji=[]
               f2= open('E:/paper/split_result/train/'+str(i)+'/'+str(j)+'.txt', 'r')
               text = f2.read().strip().decode('utf-8')
               f2.close()
               ciyuji = text.split(u',')
               #if word in ciyuji:
               if(ciyuji.__contains__(word)):
                   count=count+1
           print count
        #print word.encode('utf-8')+":"+str(count)
        f1 = open('E:/paper/df/origin.txt', 'a+')
        #f1=open('E:/paper/lda/8108/df.txt','a+')
        f1.write(word.encode('utf-8')+":"+str(count))
        f1.write('\n')
f1.close()
