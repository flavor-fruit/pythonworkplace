__author__ = 'Administrator'
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

def centreofkind(matrix):
    b=matrix.sum(axis=0)
    c=b*0.001
    return c

def classify(testArr,matrix):

    diffMat = np.tile(testArr, (8,1)) - matrix

    sqDiffMat = diffMat**2

    sqDistances = sqDiffMat.sum(axis=1)

    distances = sqDistances**0.5

    dictofkind = {'1':distances[0],'2':distances[1],'3':distances[2],'4':distances[3],'5':distances[4],'6':distances[5],'7':distances[6],'8':distances[7]}

    newdictofkind = sorted(dictofkind.iteritems(),key = lambda d:d[1],reverse =False)

    # print newdictofkind
    # print newdictofkind[0][0]
    # type(newdictofkind[0][0])
    return newdictofkind[0][0]

if __name__ == '__main__':
    start =time.clock()
    matrix = file2matrix('E:/paper/python/tfidf12/1.txt')
    textmatrix = centreofkind(matrix)
    for i in range(2,9):
        matrix = file2matrix('E:/paper/python/tfidf12/'+str(i)+'.txt')
        temp= centreofkind(matrix)
        textmatrix = np.vstack((textmatrix,temp))
    num={}
    allcount={}
    allcount[1]=allcount[2]=allcount[3]=allcount[4]=allcount[5]=allcount[6]=allcount[7]=allcount[8]=0
    for l in range(1,9):
        testmatrix = file2matrix('E:/paper/python/tvector12/'+str(l)+'.txt')
        count = 0
        for j in range(1,250):
            print int(classify(testmatrix[j,:],textmatrix))
            print allcount[int(classify(testmatrix[j,:],textmatrix))]
            allcount[int(classify(testmatrix[j,:],textmatrix))] += 1
            if((classify(testmatrix[j,:],textmatrix))==str(l)):
                count=count+1
        num[l]=count
    end = time.clock()
    f= open( 'E:/paper/python/ResOfClass/df12.txt', 'a')
    runtime=end-start
    f.write("centre's  runtime:"+str(runtime)+"Seconds")
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

