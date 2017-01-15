# -*- encoding:utf-8 -*-
import math
import csv
def tfidf_weight(dfpath, train_dfidf_path, test_dfidf_path):
    # 将原始词典词语和DF值存入word词典
    f = open(dfpath, 'r')
    # f= open('E:/paper/split_result/lda/3505/df.txt.txt', 'r')
    word = {}
    count = 0
    while True:
        # 取10用utf8 但
        line = f.readline().strip('\n').split(':')
        if len(line)>1:
            #line = line.split('\t')
            # for i in line:
            #   print i.encode('utf-8')
            count = count + 1
            # print count
            # print line[0].encode('utf-8')+':'+line[1].encode('utf-8')
            word[line[0].decode('utf-8')] = line[1]
        else:
            break
    # a=0
    # for key,value in word.items():
    #     a=a+1
    #     print a
    #     print 'key=',key.encode('utf-8'),'，value=',value.encode('utf-8')

    # 对训练集数值化（tfidf）并写入csv文件
    l = 0
    tLen = len(word)
    csvfile1 = file(train_dfidf_path, 'wb')
    for i in range(1, 9):

        writer = csv.writer(csvfile1)
        for j in range(1, 1001):
            word_in_file = {}
            f = open('E:/paper/split_result/train/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8')
            txtlist = txtlist.split(u',')

            newData = []

            if l == 0:
                data = []
                for x in range(tLen + 1):
                    data.append("X" + str(x))
                writer.writerow(data)
                l = l + 1
            for key, value in word.items():
                if not word_in_file.has_key(key):
                    sum = len(txtlist)
                    count = txtlist.count(key) * 1.0
                    word_in_file[key] = count
                    tfidf = ((count / sum) * (math.log10(8000.0 / int(value))))
                    k = (8000 / int(value))
                    newData.append(tfidf)
            newData.append(str(i))
            writer.writerow(newData)
    csvfile1.close()
        # print tfidf
    # 对测试集数值化（tf）
    k=0
    csvfile2 = file(test_dfidf_path, 'wb')
    for i in range(1, 9):

        writer = csv.writer(csvfile2)
        for j in range(1001, 1251):
            newData = []
            if k == 0:
                data = []
                for x in range(tLen + 1):
                    data.append("X" + str(x))
                writer.writerow(data)
                k = k + 1
            word_in_file = {}
            f = open('E:/paper/split_result/test/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8')
            txtlist = txtlist.split(u',')
            for key, value in word.items():
                sum = len(txtlist)
                count = txtlist.count(key) * 1.0
                tf = count / sum
                newData.append(tf)
            newData.append(str(i))
            writer.writerow(newData)
    csvfile2.close()

def bool_weight(dic_path,train_bool_path,test_bool_path):
    f = open(dic_path, 'r')

    vsm_dic = f.read().strip().decode('utf-8')
    vsm_dic = vsm_dic.split(u'\n')
    m=0
    print vsm_dic
    tLen = len(vsm_dic)
    print  "len"
    print tLen
    csvfile3 = file(train_bool_path, 'wb')
    for i in range(1, 9):
        writer = csv.writer(csvfile3)
        for j in range(1, 1001):
            word_in_file = {}
            f = open('E:/paper/split_result/train/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8')
            txtlist = txtlist.split(u',')
            l = 0
            newData = []
            if m == 0:
                data = []
                for x in range(tLen + 1):
                    data.append("X" + str(x))
                writer.writerow(data)
                m = m + 1
            for word in vsm_dic:
                if (txtlist.__contains__(word)):
                    bool_value=1
                else:
                    bool_value=0
                newData.append(bool_value)
            newData.append(str(i))
            writer.writerow(newData)
    csvfile3.close()
    k=0
    csvfile4 = file(test_bool_path, 'wb')
    for i in range(1, 9):

        writer = csv.writer(csvfile4)
        for j in range(1001, 1251):
            newData = []
            if k == 0:
                data = []
                for x in range(tLen + 1):
                    data.append("X" + str(x))
                writer.writerow(data)
                k = k + 1
            word_in_file = {}
            f = open('E:/paper/split_result/test/' + str(i) + '/' + str(j) + '.txt', 'r')
            txtlist = f.read().strip().decode('utf-8')
            txtlist = txtlist.split(u',')
            for word in vsm_dic:
                if (txtlist.__contains__(word)):
                    bool_value=1
                else:
                    bool_value=0

                newData.append(bool_value)
            newData.append(str(i))
            writer.writerow(newData)
    csvfile4.close()


if __name__ == '__main__':
    tfidf_weight('E:/paper/lda/8180/df.txt', 'E:/paper/lda/8180/tfidf/train.csv', 'E:/paper/lda/8180/tfidf/test.csv')
    bool_weight('E:/paper/lda/8180/8180.txt',  'E:/paper/lda/8180/bool/train.csv',  'E:/paper/lda/8180/bool/test.csv')