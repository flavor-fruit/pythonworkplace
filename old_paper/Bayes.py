__author__ = 'Administrator'
import math
import numpy as np
import time
def file2matrix(filename):
    fr = open(filename,'r')
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,3789))        #prepare matrix to return
    #classLabelVector = []                       #prepare labels return
    index = 0
    for line in arrayOLines:
        if not line.strip():
            continue
        else:
           listFromLine = line.strip().split('\t')

           returnMat[index,:] = listFromLine[0:3789]

           index += 1
    return returnMat

def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = math.zeros(numWords);
    p1Num = math.zeros(numWords);
    p0Denom = 0.0;
    p1Denom = 0.0;
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
             p1Num += trainMatrix[i]
             p1Denom += sum(trainMatrix[i])
        else:
              p0Num += trainMatrix[i]
              p0Denom += sum(trainMatrix[i])
              p1Vect = p1Num / p1Denom
              p0Vect = p0Num / p1Denom
    return p0Vect, p1Vect, pAbusive

def trainNB0(trainMatrix):
    pNum =np.ones(3789)
    pDenom = 2.0
    for i in range(len(trainMatrix)):
        pNum+=trainMatrix[i]
        pDenom+= sum(trainMatrix[i])
    pVect = np.log(pNum / pDenom )
    return pVect

def testingNB(matrix,tvector):
    p={}
    for i in range(8):
        p[i+1]=sum(matrix[i]*tvector)
    p = sorted(p.iteritems(),key = lambda d:d[1],reverse =True)

    #print p
    return  p[0][0]



if __name__ == '__main__':
    start =time.clock()
    pmat = np.zeros((8,3789))
    for i in range(1,9):
        matrix = file2matrix('E:/paper/python/tfidf12/'+str(i)+'.txt')
        pmat[i-1]=trainNB0(matrix)

    num={}
    allcount={}
    allcount[1]=allcount[2]=allcount[3]=allcount[4]=allcount[5]=allcount[6]=allcount[7]=allcount[8]=0
    for l in range(1,9):
        testmatrix = file2matrix('E:/paper/python/tvector12/'+str(l)+'.txt')
        count=0
        for j in range(1,250):
            allcount[testingNB(pmat, testmatrix[j])] += 1
            if(testingNB(pmat,testmatrix[j])==l):
                count=count+1
        num[l]=count
    end = time.clock()
    f= open( 'E:/paper/python/ResOfClass/df12.txt', 'a')
    runtime=end-start
    f.write("Bayes's  runtime:"+str(runtime)+"Seconds")
    f.write('\n')


    f.write("m"+'\n')
    for k in range(1,9):
      f.write(str(num[k]/250.0)+'\t')
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