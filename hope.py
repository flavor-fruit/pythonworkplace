__author__ = 'user'

import numpy as np
def getData(path):
    file1 = open(path,'r')
    trainX = []
    i = 0
    for line in file1.readlines():

        data = []
        segList = line.strip("\n").split(",")
        for w in range(0,len(segList)):
            data.append(segList[w])
        trainX.append(data)
    return trainX

import pandas as pd
from sklearn import svm
number=250
print 'This is %d'%number


#trainX = getData("E:\\paper\\mi\\%d\\train_tfidf.csv"%number)
trainX=getData("E:\\paper\\sougou\\lda_topic\\%d\\train.csv"%number)
trainX=np.array(trainX)
trainY=[i / 1000 + 1 for i in range(8000)]

trainY = np.array(trainY)
#testX = getData("E:\\paper\\mi\\%d\\test_tfidf.csv"%number)
testX=getData("E:\\paper\\sougou\\lda_topic\\%d\\test.csv"%number)
testX = np.array(testX)
testY=[i / 250 + 1 for i in range(2000)]
testY = np.array(testY)
import scipy
titles = ['LinearSVC (linear kernel)',
          'SVC with polynomial (degree 3) kernel',
          'SVC with RBF kernel',
          'SVC with Sigmoid kernel']
# x = [[0, 0], [1, 1]]
# y = [0.5, 1.5]
# clf = svm.SVR()
# clf.fit(x, y)
clf_linear = svm.SVC(kernel='linear').fit(trainX, trainY)
print clf_linear
predicted = clf_linear.predict(testX)
import time
# print predicted
from sklearn import metrics
start =time.clock()
predicted = clf_linear.predict(testX)
end = time.clock()
from sklearn.externals import joblib
#joblib.dump(clf_linear, 'svm.pkl')
print(metrics.classification_report(testY, predicted))
print('Running time: %s Seconds'%(end-start))