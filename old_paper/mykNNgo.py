#-*_coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from numpy import *
import numpy as np
import mykNN
import time
import matplotlib
import matplotlib.pyplot as plt

# kNN.datingClassTest()

#def classifyText():
if __name__ == '__main__':
     start =time.clock()
     #DataMats= mykNN.file2matrix('vector.txt')
     DataMats= mykNN.file2matrix('E:/paper/python/tfidf12/1.txt')

     for i in range(2,9):
        smallData = mykNN.file2matrix('E:/paper/python/tfidf12/'+str(i)+'.txt')
        DataMats=np.vstack((DataMats,smallData))
     textLabels=[1]*1000+[2]*1000+[3]*1000+[4]*1000+[5]*1000+[6]*1000+[7]*1000+[8]*1000
     #normMat, ranges, minVals = mykNN.autoNorm(DataMats)

     #knn�в����ı任
     f= open( 'E:/paper/python/ResOfClass/df12.txt', 'a')
     for k in range(5,56,10):
         f.write("KNN's  k:"+str(k)+'\n')
         num={}
         allcount={}
         allcount[1]=allcount[2]=allcount[3]=allcount[4]=allcount[5]=allcount[6]=allcount[7]=allcount[8]=0
         for l in range(1,9):
            testmatrix = mykNN.file2matrix('E:/paper/python/tvector12/'+str(l)+'.txt')
            count = 0
            for j in range(1,250):
                inArr=testmatrix[j]
                #print mykNN.classify0(inArr,DataMats,textLabels,3)
                #print l
               # print mykNN.classify0((inArr-minVals)/ranges,normMat,textLabels,3)
               # if(mykNN.classify0((inArr-minVals)/ranges,normMat,textLabels,4)!=l):
                allcount[mykNN.classify0(inArr,DataMats,textLabels,k)] += 1
                if(mykNN.classify0(inArr,DataMats,textLabels,k)==l):
                    count=count+1
            num[l]=count
         end = time.clock()

         runtime=end-start
         f.write("KNN's  runtime:"+str(runtime)+"Seconds")
         f.write('\n')
         f.write("m"+'\n')
         for m in range(1,9):
           f.write(str(num[m]/250.0)+'\t')
         f.write('\n')
         avem=(num[1]/250.0+ num[2]/250.0+ num[3]/250.0+ num[4]/250.0+ num[5]/250.0+ num[6]/250.0+ num[7]/250.0+ num[8]/250.0)*1.0/8
         f.write("avem"+str(avem))
         f.write('\n')
         f.write("q"+'\n')
         for p in range(1,9):
           f.write(str(num[p]*1.0/allcount[p])+'\t')
         f.write('\n')
         aveq=(num[1]*1.0/allcount[1]+num[2]*1.0/allcount[2]+num[3]*1.0/allcount[3]+num[4]*1.0/allcount[4]+num[5]*1.0/allcount[5]+num[6]*1.0/allcount[6]+num[7]*1.0/allcount[7]+num[8]*1.0/allcount[8])*1.0/8
         f.write("aveq"+str(aveq)+'\n')

     f.close()




