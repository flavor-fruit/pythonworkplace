#-*- encoding:utf-8 -*-
import math
def tfidf_weight(dfpath,train_dfidf_path,test_dfidf_path):
    #将原始词典词语和DF值存入word词典
    f= open('E:/paper/df/.txt.txt', 'r')
    #f= open('E:/paper/split_result/lda/3505/df.txt.txt', 'r')
    word={}
    count =0
    while True:
        #取10用utf8 但
        line =f.readline().decode('utf-8')
        if line:
           line = line.split('\t')
           # for i in line:
           #   print i.encode('utf-8')
           count=count+1
           #print count
           #print line[0].encode('utf-8')+':'+line[1].encode('utf-8')
           word[line[0]]=line[1]
        else:
            break
    # a=0
    # for key,value in word.items():
    #     a=a+1
    #     print a
    #     print 'key=',key.encode('utf-8'),'，value=',value.encode('utf-8')

    #对训练集数值化（tfidf）并写入csv文件
    import csv

    for i in range(1,9):
        csvfile = file('E:/paper/tfidf/train/ /' + str(i) + '.csv', 'wb')
        writer = csv.writer(csvfile)
        fw= open( 'E:/paper/tfidf/train/ /'+str(i)+'.txt', 'a')
        for j in range(1,1001):
             word_in_file={}
             f= open( 'E:/paper/split_result/train/'+str(i)+'/'+str(j)+'.txt', 'r')
             txtlist = f.read().strip().decode('utf-8')
             txtlist=txtlist.split(u',')
             l=0
             newData=[]
             for key,value in word.items():
                  if not word_in_file.has_key(key):
                     sum = len(txtlist)
                     count = txtlist.count(key)*1.0
                     word_in_file[key]=count
                     tfidf =((count/sum)*(math.log10(8000.0/int(value))))
                     k=(8000/int(value))
                     newData.append(tfidf)
        newData.append(str(i))
        writer.writerow(newData)
        csvfile.close()
                     #print tfidf
    #对测试集数值化（tf）
    for i in range(1,9):
        csvfile = file('E:/paper/tfidf/test/ /' + str(i) + '.csv', 'wb')
        writer = csv.writer(csvfile)
        for j in range(1001,1251):
             newData = []
             word_in_file={}
             f= open( 'E:/paper/split_result/test/'+str(i)+'/'+str(j)+'.txt', 'r')
             txtlist = f.read().strip().decode('utf-8')
             txtlist=txtlist.split(u',')
             for key,value in word.items():
                 sum = len(txtlist)
                 count = txtlist.count(key)*1.0
                 tf=count/sum
                 newData.append(tf)
        newData.append(str(i))
        writer.writerow(newData)
        csvfile.close()

