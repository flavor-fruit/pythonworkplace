__author__ = 'user'
from sklearn import metrics
import numpy as np
import time
from sklearn import svm
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
        for w in range(1,len(segList)):
            data.append(segList[w])
        trainX.append(data)
    return trainX

import pandas as pd

number_list = [10,20,200]
ex_path = 'E:\paper\lda_word'
for number in number_list:
    print 'This is %d' % number

    trainX = getData("%s\\%d\\train_tfidf.csv" %(ex_path,number))
    # trainX = getData('E:\\paper\\undescend\\train_tfidf.csv')
    trainX = np.array(trainX)
    trainY = [i / 1000 + 1 for i in range(8000)]

    trainY = np.array(trainY)
    # testX = getData("E:\\paper\\lda_desend\\%d\\test_tfidf.csv"%number)
    testX = getData('%s\\%d\\test_tfidf.csv'%(ex_path,number))
    testX = np.array(testX)
    testY = [i / 250 + 1 for i in range(2000)]
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

    # print predicted


    start = time.clock()
    predicted = clf_linear.predict(testX)
    end = time.clock()
    from sklearn.externals import joblib

    # joblib.dump(clf_linear, 'svm.pkl')
    print(metrics.classification_report(testY, predicted))
    print('Running time: %s Seconds' % (end - start))