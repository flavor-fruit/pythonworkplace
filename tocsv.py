__author__ = 'Administrator'
# -*- coding: utf-8 -*-
import csv
path = "E:\\paper\\python\\df\\tvector6\\"
newFile = open("E:\\paper\\newjob\\csv\\testVector\\df6.csv",'wb')
writer = csv.writer(newFile)
j=0
for i in range(1,9):
    newPath = path+str(i)+".txt"
    myFile = open(newPath,'r')

    count = 0
    for line in myFile.readlines():
        count = count+1;
        #if line.strip("\n").strip(" ")=="":
           # continue
        #seg_list = line.strip("\n").strip("  ").split("  ")

        seg_list = line.strip("\n").strip().split("\t")
        print seg_list
        tLen = len(seg_list)
        print tLen
        newData = []
        if j==0:
            data = []
            for x in range(tLen+1):
                data.append("X"+str(x))
            writer.writerow(data)
            j = j+1
        for w in seg_list:
            newData.append(w)
        # print newData
        newData.append(str(i))
        writer.writerow(newData)
    print count
newFile.close()
