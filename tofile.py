# -*- coding: utf-8 -*-
#encoding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')



f1=open("E:/paper/testdoc.txt",'a')
for i in range(1, 9):
    for j in range(1001, 1251):
        f = open('E:/paper/split_result/test/' + str(i) + '/' + str(j) + '.txt', 'r')
        txtlist = f.read().strip().replace(',',' ').decode('utf-8')
        f1.write(txtlist)
        f1.write("\n")
        f.close()
f1.close()