__author__ = 'user'
import time
start =time.clock()
import numpy as np
def getData(path):
    file1 = open(path,'r')
    trainX = []
    trainY = []
    i = 0
    for line in file1.readlines():
        if i==0:
            i = i+1
            continue;
        data = []
        segList = line.strip("\n").split(",")
        for w in range(len(segList)-1):
            data.append(segList[w])
        trainX.append(data)
        trainY.append(segList[len(segList)-1])
    return trainX,trainY
#KNN Classifier
from sklearn.naive_bayes import MultinomialNB
trainX ,trainY= getData("E:\\paper\\lda\\8180\\tfidf\\train.csv")
trainX = np.array(trainX)
trainY = np.array(trainY)
testX,testY = getData("E:\\paper\\lda\\8180\\tfidf\\test.csv")
testX = np.array(testX)
testY = np.array(testY)
clf = MultinomialNB(alpha = 0.01)
clf.fit(trainX,trainY);
end = time.clock()
pred = clf.predict(testX);
from sklearn import metrics
print(metrics.classification_report(testY, pred))
print('Running time: %s Seconds'%(end-start))