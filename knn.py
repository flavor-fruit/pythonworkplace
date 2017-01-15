__author__ = 'user'
import time
start =time.clock()
import numpy as np
def getData(path):
    file1 = open(path,'r')
    trainX = []
    i = 0
    for line in file1.readlines():
        if i==0:
            i = i+1
            continue;
        data = []
        segList = line.strip("\n").split(",")
        for w in range(len(segList)):
            data.append(segList[w])
        trainX.append(data)
    return trainX
#KNN Classifier
number=6555
from sklearn.neighbors import KNeighborsClassifier
trainX= getData("E:\\paper\\df\\%d\\train_tfidf.csv"%number)
trainX = np.array(trainX)
trainY=[i / 1000 + 1 for i in range(8000)]
# print "label"
# print trainY
testX = getData("E:\\paper\\df\\%d\\test_tfidf.csv"%number)
testX = np.array(testX)
testY=[i / 250 + 1 for i in range(2000)]
# print len(testY)
# print testY
test_dicY = {}
for i in range(len(testY)):
    if test_dicY.has_key(testY[i]):
        test_dicY[testY[i]] += 1
    else:
        test_dicY[testY[i]] = 1
print test_dicY
knnclf = KNeighborsClassifier()#default with k=5
knnclf.fit(trainX,trainY)
end = time.clock()
pred = knnclf.predict(testX);
from sklearn import metrics
# print len(pred)
pre_dic = {}
for i in range(len(pred)):
    if pre_dic.has_key(pred[i]):
        pre_dic[pred[i]] += 1
    else:
        pre_dic[pred[i]] = 1
print pre_dic
# print pred
print(metrics.classification_report(testY, pred))
print('Running time: %s Seconds'%(end-start))



