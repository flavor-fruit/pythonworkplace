# -*- coding: utf-8 -*-
#encoding=utf-8
__author__ = 'user'
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
import time
import math

def is_chinese(uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
                return True
        else:
                return False

import MySQLdb
#数据库连接

def connectDataBase():
    conn= MySQLdb.connect(
        host='localhost',
        port = 3306,
        user='root',
        passwd='',
        db ='patentdata',
        )
    conn.set_character_set('utf8')
    cur = conn.cursor()
    cur.execute('SET NAMES utf8;')
    cur.execute('SET CHARACTER SET utf8;')
    cur.execute('SET character_set_connection=utf8;')

    return conn,cur

# 解析json数据并保存到数据库中

import json

def readDataFromJson():
    JsonFile = open("C:\\Users\\user\\Desktop\\items.json",'r')
    row = 0
    for eachline in JsonFile:
        conn,cur = connectDataBase()
        line = eachline.strip().strip("\n").decode('utf-8')
        js = None
        try:
            js = json.loads(line)
        except Exception,e:
            print "bad line"
        rows = row
        title = js['title'].strip()
        abstract = js['abstract'].strip()
        newAbstract = ""
        for i in range(len(abstract)):
            if(is_chinese(abstract[i])):
                continue
            newAbstract = newAbstract+abstract[i]
        year = js['year'].strip()
        author = js['author'].strip()
        sqli="insert into dervent1 values(%s,%s,%s,%s,%s)"
        cur.execute(sqli,(rows,title,newAbstract,year,author))
        row = row+1
        # print rows
        cur.close()
        conn.commit()
        conn.close()
    print "json读取成功，并写入数据库！"
# readDataFromJson()
# 读取数据库中的数据将摘要信息放到abstract.txt文件中
def readFromDataBase(filePath,value):
    abstractFile = open(filePath,'w')
    conn,cur = connectDataBase()
    linesNum = cur.execute("select rows from patent_rest_use_lsi")
    # print linesNum
    lines = cur.fetchmany(linesNum)

    for line in lines:
        abstractFile.write(str(line[value]))
        # abstractFile.write(":")
        # abstractFile.write(line[value+1])
        abstractFile.write("\n")
    abstractFile.close()
    cur.close()
    conn.commit()
    conn.close()
    print "complete!"
# readFromDataBase("E:\\derventData\\rows.txt",0)
from gensim import corpora, models, similarities
import logging

def dealWithAbstract(filepath):
    sourceFile = open(filepath,'r')
    titleData = []
    for line in sourceFile:
        titleData.append(line.strip("\n"))
    # print titleData[0:10]
    texts_lower = [[word for word in document.lower().split()] for document in titleData]
    print texts_lower[0]
    from nltk.tokenize import word_tokenize
    texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in titleData]
    print texts_tokenized[0]
    from nltk.corpus import stopwords
    english_stopwords = stopwords.words('english')
    texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in texts_tokenized]
    print texts_filtered_stopwords[0]
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']','&','!', '*','@','#','$', '%','-']
    texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]
    print texts_filtered[0]
    # from nltk.stem.lancaster import LancasterStemmer
    # st = LancasterStemmer()
    from nltk.stem.porter import PorterStemmer
    porter_stemmer = PorterStemmer()
    texts_stemmed = [[porter_stemmer.stem(word) for word in docment] for docment in texts_filtered]
    print texts_stemmed[0]
    all_stems = sum(texts_stemmed, [])
    stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
    texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]
    #
    dic_check = []
    for doc in texts:
        for i in range(len(doc)):
            if dic_check.__contains__(doc[i]):
                continue
            else:
                dic_check.append(doc[i])
    print len(dic_check)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    dictionary = corpora.Dictionary(texts)
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "字典的长度为%d"%(len(dictionary))
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    # index = similarities.MatrixSimilarity(lsi[corpus])
    print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&77"
    create_new_dic = []
    for i in range(len(corpus_tfidf)):
        dic_word = {}
        count = 0
        temp_list = []
        corpus_tfidf[i] = [(0,123),(1,123)]
        # for j in range(len(corpus_tfidf[i])):
        #     corpus_tfidf[i][j] = []
        #     if corpus_tfidf[i][j][1]<0.1:
        #         corpus_tfidf[i][j] = []
        #         # temp_list.append(dictionary[word[0]])
        #         count += 1
        #     dic_word[word[0]] = word[1]
        # create_new_dic.append(temp_list)
        # if count==0:
        #     print "hahha"
    selected_dictionary = corpora.Dictionary(create_new_dic)
    selected_corpus = [selected_dictionary.doc2bow(text) for text in texts]
    test_selec_cor = [selected_dictionary.doc2bow(text) for text in create_new_dic]
    selected_tfidf = models.TfidfModel(selected_corpus)
    test_selec_tfidf = models.TfidfModel(test_selec_cor)
    selected_corpus_tfidf = selected_tfidf[selected_corpus]
    test_selec_cor_tfidf = test_selec_tfidf[test_selec_cor]
    num = 0
    num1 = 0
    for c in corpus_tfidf:
        num += 1
        if num==3:
            break
        print c
    for c in selected_corpus_tfidf:
        num1 += 1
        if num1==3:
            break
        print c
    num2 = 0
    for c in test_selec_cor_tfidf:
        num2 += 1
        if num2==3:
            break
        print c
        # print sorted(dic_word.iteritems(),key=lambda asd:asd[1],reverse=True)
        # print c
    # 对计算得到的tf-idf值重新排序,并按照比例进行特征选择
    # create_new_dic = []
    # rate = 0.11
    # for doc in corpus_tfidf:
    #     temp_dic = {}
    #     for i in range(len(doc)):
    #         temp_dic[doc[i][0]] = doc[i][1]
    #     temp_dic = sorted(temp_dic.iteritems(),key=lambda asd:asd[1],reverse=True)
    #     temp_list = []
    #     new_len = int(len(temp_dic)*rate)
    #     if new_len==0:
    #         new_len = 1
    #     for i in range(new_len):
    #         temp_list.append(dictionary[temp_dic[i][0]])
    #     create_new_dic.append(temp_list)
    # selected_dictionary = corpora.Dictionary(create_new_dic)
    # selected_corpus = [selected_dictionary.doc2bow(text) for text in texts]
    # selected_tfidf = models.TfidfModel(selected_corpus)
    # selected_corpus_tfidf = selected_tfidf[selected_corpus]
    # 以上为排序及特征选择代码
    print "特征选择前的字典长度%d"%(len(dictionary))
    print "特征选择后的字典长度%d"%(len(selected_dictionary))
    return test_selec_cor_tfidf,selected_dictionary,create_new_dic,titleData
#
dealWithAbstract("E:\\derventData\\title.txt")
# 将数据表示成矩阵形式
# import gensim.models.word2vec as word2vec
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
def Train_docVector():
    corpus_tfidf,dictionary,texts,titleData = dealWithAbstract("E:\\derventData\\abstract_lda.txt")
    corpora_documents = []
    for i,item_text in enumerate(texts):
        document = TaggedDocument(words=item_text,tags=[i])
        corpora_documents.append(document)
    model = Doc2Vec(size=50,min_count=1,iter=10)
    model.build_vocab(corpora_documents)
    model.train(corpora_documents)
    print model.docvecs
# Train_docVector()
def Train_wordVector():
    corpus_tfidf,dictionary,texts,titleData = dealWithAbstract("E:\\derventData\\abstract_lsi.txt")
    wordVecModel = models.Word2Vec(texts,size=100,window=5,min_count=1,workers=4)
    textsMatrix = []
    initVector = np.zeros((1,100))[0]
    terms = wordVecModel.vocab.keys()
    count = 0
    # 对词向量采用tf-idf加权计算
    for doc in corpus_tfidf:
        initVector = np.zeros((1,100))[0]
        for i in range(len(doc)):
            if terms.__contains__(doc[i][0]):
                initVector += wordVecModel[doc[i][0]]*doc[i][1]
        textsMatrix.append(initVector/(len(doc)*1.0))

    #对词向量直接取平均值
    # for doc in texts:
    #     count = count+1
    #     initVector = np.zeros((1,100))[0]
    #     for i in range(len(doc)):
    #         # print doc[i]
    #         if(terms.__contains__(doc[i])):
    #             initVector += wordVecModel[doc[i]]
    #     textsMatrix.append(initVector/(len(doc)*1.0))
    # print dictionary[0]
    # print wordVecModel.vocab[0]
    # print wordVecModel[dictionary[0]]
    print "处理了%d篇文档"%(count)
    textsMatrix = np.array(textsMatrix)
    return textsMatrix
# Train_wordVector()
def convertToVector(filePath,num_topics,corpus_tfidf,dictionary,texts,titledata):
    # corpus_tfidf,dictionary,texts,titleData = dealWithAbstract(filePath)
    Lda_topic0_dic, Lda_topic1_dic, texts, titledata, topic_result,selected_corpus_tfidf = testDataWithLsi(filePath,num_topics,corpus_tfidf,dictionary,texts,titledata)
    # topic_result = testDataWithHdp("E:\\derventData\\title.txt")
    dataMatrix = []
    for document in topic_result:
        dic = {}
        vector = []
        if len(document)<num_topics:
            # print "wocao"
            # print document
            temp_dic = {}
            for i in range(num_topics):
                temp_dic[i] = 0.0
            for word in document:
                temp_dic[word[0]] = word[1]
            for word in temp_dic.keys():
                vector.append(temp_dic[word])
            # print "wocao"
            # print vector
            # print document
        else:
            for word in document:
                vector.append(word[1])
        dataMatrix.append(vector)

    # 对tf-idf进行向量化
    # for document in selected_corpus_tfidf:
    #     dic = {}
    #     vector = []
    #     for word in document:
    #         dic[word[0]] = word[1]
    #     for key in range(len(dictionary)):
    #         if dic.has_key(key):
    #             vector.append(dic[key])
    #         else:
    #             vector.append(0)
    #     dataMatrix.append(vector)
    # 对向量进行l2归一化，目的是为了使得近似欧氏距离与余弦距离
    dataMatrix = np.array(dataMatrix)
    # dataMatrix.s
    print dataMatrix.dtype
    # print dataMatrix

    for i in range(len(dataMatrix)):
        dataMatrix[i] = dataMatrix[i]/math.sqrt(sum(dataMatrix[i]**2))
    return dataMatrix

def testDataWithHdp(filepath,corpus_tfidf,dictionary,texts,titledata):
    print "hdp starts"
    # corpus_tfidf,dictionary,texts,titledata = dealWithAbstract(filepath)
    print '\n\nUSE WITH CARE--\nHDA Model:'
    hda = models.HdpModel(corpus_tfidf, id2word=dictionary)
    topic_result = [a for a in hda[corpus_tfidf]]
    # return topic_result
    # pprint(topic_result)
    dic_distribution = {}
    for doc in topic_result:
        for i in range(len(doc)):
            if dic_distribution.has_key(doc[i][0]):
                dic_distribution[doc[i][0]] += 1
            else:
                dic_distribution[doc[i][0]] = 1
    dic = sorted(dic_distribution.iteritems(),key=lambda asd:asd[1],reverse=True)
    for i in range(len(dic)):
        print "%s : %d"%(dic[i][0],dic[i][1])
    print 'HDA Topics:'
    # print hda.print_topics(num_topics=100,num_words=5)
    # print hda.
# testDataWithHdp("E:\\derventData\\title.txt")
def testDataWithLsi(filepath, num_topics,corpus_tfidf,dictionary,texts,titledata):
    print "lsi starts!"
    start_time = time.time()
    # corpus_tfidf,dictionary,texts,titledata = dealWithAbstract(filepath)
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    topic_result = [a for a in lsi[corpus_tfidf]]
    # pprint(topic_result)
    print "lsi topics"
    # pprint(lsi.print_topics(num_topics=2,num_words=10))
    Lsi_topic0_dic = []
    Lsi_topic1_dic = []
    for i in range(len(texts)):
        ml_course = texts[i]
        ml_bow = dictionary.doc2bow(ml_course)
        ml_lsi = lsi[ml_bow]
        if (len(ml_lsi)==1 and ml_lsi[0][0]==0) or(len(ml_lsi)==2 and ml_lsi[0][1]>ml_lsi[1][1]):
            Lsi_topic0_dic.append(i)
        else:
            Lsi_topic1_dic.append(i)
    # print ml_lsi
    print "lsi ends!"
    end_time = time.time()
    print "lsi costs %.3f s"%(end_time-start_time)
    selectec_corpus_tfidf = []
    return Lsi_topic0_dic,Lsi_topic1_dic,texts,titledata, topic_result,selectec_corpus_tfidf
# testDataWithLsi("E:\\derventData\\title.txt",20)
def testDataWithLDA(filepath, num_topics,corpus_tfidf,dictionary,texts,titledata):
    print "Lda start!"
    start_time = time.time
    # corpus_tfidf,dictionary,texts,titledata = dealWithAbstract(filepath)
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                            alpha=0.01, eta=0.01, minimum_probability=0.001,
                            update_every = 1, chunksize = 100, passes = 1)
    topic_result = [a for a in lda[corpus_tfidf]]
    # pprint(topic_result)
    print len(dictionary)
    print "lda topics"
    selected_corpus_tfidf = []
    # 提取文本-主题-词特征
    # rate = 0.8
    # new_texts = []
    # for doc_topic in topic_result:
    #     dic_percent = {}
    #     topic_sum = 0.0
    #     for i in range(len(doc_topic)):
    #         topic_sum += doc_topic[i][1]
    #     for i in range(len(doc_topic)):
    #         topic_id = doc_topic[i][0]
    #         topic_percent = doc_topic[i][0]/topic_sum
    #         # print "topic",topic_id
    #         # lda.
    #         print topic_percent
    #         terms = lda.get_topic_terms(topic_id,topn=100000000)
    #         single_text = []
    #         for j in range(int(len(terms)*rate*topic_percent)):
    #             single_text.append(dictionary[terms[j][0]])
    #         new_texts.append(single_text)
    # selected_dictionary = corpora.Dictionary(new_texts)
    # selected_corpus = [selected_dictionary.doc2bow(text) for text in texts]
    # selected_tfidf = models.TfidfModel(selected_corpus)
    # selected_corpus_tfidf = selected_tfidf[selected_corpus]
            # print m[0][0]
            # print len(m)
            # print dictionary[m[0][0]]
            # pprint(m)
            # pprint(lda.show_topic(topic_id,topn=100000000))
            # print dictionary[1346]
    Lda_topic0_dic = []
    Lda_topic1_dic = []
    for i in range(len(texts)):
        ml_course = texts[i]
        ml_bow = dictionary.doc2bow(ml_course)
        ml_lda = lda[ml_bow]
        # print "*******"
        # print ml_lda
        if (len(ml_lda)==1 and ml_lda[0][0]==0) or (len(ml_lda)==2 and ml_lda[0][1]>ml_lda[1][1]):
            Lda_topic0_dic.append(i)
        else:
            Lda_topic1_dic.append(i)
    print "lda ends!"
    end_time = time.time()
    # print "lDA costs %.3f s"%(end_time-start_time)
    # return Lda_topic0_dic, Lda_topic1_dic, texts, titledata, topic_result
    return Lda_topic0_dic, Lda_topic1_dic, texts, titledata, topic_result,selected_corpus_tfidf
# 聚类开始
# corpus_tfidf,dictionary,texts,titledata = dealWithAbstract("E:\\derventData\\title.txt")
# testDataWithLDA("E:\\derventData\\title.txt",20,corpus_tfidf,dictionary,texts,titledata)
from sklearn.cluster import KMeans
#
def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d
#

# 分别使用LSI和LDA对摘要信息进行主题提取，并根据主题筛选摘要，把筛选出的摘要存储到新的表中，剩下的数据也存储到新表中
def abstract_deleted_and_new_rest(filePath,num_topics):
    abstract_topic0_dic_lsi,abstract_topic1_dic_lsi,texts_lsi,titledata_lsi, topic_result_lsi= testDataWithLsi(filePath,num_topics)
    len_topic0 = len(abstract_topic1_dic_lsi)
    len_topic00 = len(abstract_topic0_dic_lsi)
    print "lsi 可以删除的文本个数为:%d"%(len_topic0)
    print "它们是如下文本："
    for i in range(len(abstract_topic1_dic_lsi)):
        print titledata_lsi[abstract_topic1_dic_lsi[i]]

    abstract_topic0_dic_lda,abstract_topic1_dic_lda,texts_lda,titledata_lda, topic_result_lda = testDataWithLDA(filePath,num_topics)

    len_topic1 = len(abstract_topic1_dic_lda)
    len_topic10 = len(abstract_topic0_dic_lda)
    print "lda可以删除的文本数为:%d"%(len_topic1)
    count = 0;
    for i in range(len_topic0):
        if(abstract_topic1_dic_lda.__contains__(abstract_topic1_dic_lsi[i])):
            count = count+1
    print "$$$$$$$$$$$$$$"
    print "lda和lsi共同删除的文本数为：%d"%(count)
    print "将删除的文本写入数据库。。。"
    print "lsi删除的文本写入patent_delete_use_lsi..."
    conn,cur = connectDataBase()
    for i in range(len_topic0):
            conn,cur = connectDataBase()
            # line = eachline.strip().strip("\n").decode('utf-8')
            # js = None
            # try:
            #     js = json.loads(line)
            # except Exception,e:
            #     print "bad line"
            rows = abstract_topic1_dic_lsi[i]
            # title = js['title'].strip()
            abstract = titledata_lsi[abstract_topic1_dic_lsi[i]]
            sqli="insert into patent_delete_use_lsi values(%s,%s)"
            cur.execute(sqli,(rows,abstract))
            # print rows
            cur.close()
            conn.commit()
            conn.close()
    print "patent_delete_use_lsi writing success!"
    print "lda删除的文本写入patent_delete_use_lda..."
    conn,cur = connectDataBase()
    for i in range(len_topic1):
            conn,cur = connectDataBase()
            # line = eachline.strip().strip("\n").decode('utf-8')
            # js = None
            # try:
            #     js = json.loads(line)
            # except Exception,e:
            #     print "bad line"
            rows = abstract_topic1_dic_lda[i]
            # title = js['title'].strip()
            abstract = titledata_lsi[abstract_topic1_dic_lda[i]]
            sqli="insert into patent_delelte_use_lda values(%s,%s)"
            cur.execute(sqli,(rows,abstract))
            # print rows
            cur.close()
            conn.commit()
            conn.close()
    print "patent_delete_use_lda writing success!"

    print "lda留下的文本写入patent_rest_use_lda..."
    conn,cur = connectDataBase()
    for i in range(len_topic10):
            conn,cur = connectDataBase()
            # line = eachline.strip().strip("\n").decode('utf-8')
            # js = None
            # try:
            #     js = json.loads(line)
            # except Exception,e:
            #     print "bad line"
            rows = abstract_topic0_dic_lda[i]
            # title = js['title'].strip()
            abstract = titledata_lsi[abstract_topic0_dic_lda[i]]
            sqli="insert into patent_rest_use_lda values(%s,%s)"
            cur.execute(sqli,(rows,abstract))
            # print rows
            cur.close()
            conn.commit()
            conn.close()
    print "patent_rest_use_lda writing success!"
    print "lsi留下的文本写入patent_rest_use_lsi..."
    conn,cur = connectDataBase()
    for i in range(len_topic00):
            conn,cur = connectDataBase()
            # line = eachline.strip().strip("\n").decode('utf-8')
            # js = None
            # try:
            #     js = json.loads(line)
            # except Exception,e:
            #     print "bad line"
            rows = abstract_topic0_dic_lsi[i]
            # title = js['title'].strip()
            abstract = titledata_lsi[abstract_topic0_dic_lsi[i]]
            sqli="insert into patent_rest_use_lsi values(%s,%s)"
            cur.execute(sqli,(rows,abstract))
            # print rows
            cur.close()
            conn.commit()
            conn.close()
    print "patent_rest_use_lsi writing success!"
# abstract_deleted_and_new_rest()
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
def dimention_reduced_use_pca():
    from sklearn import metrics
    max_i = 0
    max_j = 0
    max_value = 0.0
    corpus_tfidf,dictionary,texts,titledata = dealWithAbstract("E:\\derventData\\abstract_lsi.txt")
    for i in range(2,30):
        dataMatrix = convertToVector("E:\\derventData\\abstract_lsi.txt",i,corpus_tfidf,dictionary,texts,titledata)
        # dataMatrix = Train_wordVector()
        # pca=PCA(n_components=3)
        # dimention_reduced_newData = pca.fit_transform(dataMatrix)
        N = 400
        centers = 6
        # newData = convertToVector("E:\\derventData\\title.txt")
        for j in range(2,30):
            cls = KMeans(n_clusters=j, init='k-means++')
            y_hat = cls.fit_predict(dataMatrix)

            # 轮廓系数
            if len(np.unique(y_hat))==1:
                continue
            pro_coeffi = metrics.silhouette_score(dataMatrix, cls.labels_, metric="euclidean")
            if pro_coeffi>max_value:
                max_i = i
                max_j = j
                max_value = pro_coeffi
            print "主题数为%d,类簇数为%d，轮廓系数为%f"%(i,j,pro_coeffi)
    print "最好的主题数为%d,最好的类簇数为%d，轮廓系数为%f"%(max_i,max_j,max_value)

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(20,21), facecolor='w')
    # plt.subplot(421)
    plt.title(u'原始数据')
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    print "the data"
    # print dimention_reduced_newData
    # ax.scatter(dimention_reduced_newData[:, 0], dimention_reduced_newData[:, 1],dimention_reduced_newData[:, 2], c=y_hat, s=30, cmap=cm, edgecolors='none')
    print "jhhh"
    # print data[:,0]
    # x1_min, x2_min,x3_min = np.min(dimention_reduced_newData, axis=0)
    # x1_max, x2_max,x3_max = np.max(dimention_reduced_newData, axis=0)
    # x1_min, x1_max = expand(x1_min, x1_max)
    # x2_min, x2_max = expand(x2_min, x2_max)
    # x3_min, x3_max = expand(x3_min, x3_max)
    ax.set_zlabel('Z') #坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    # plt.grid(True)
    plt.show()
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
def clustering_use_DBSCAN():
    corpus_tfidf,dictionary,texts,titledata = dealWithAbstract("E:\\derventData\\abstract_lda.txt")
    max_value = 0.0
    max_i = 0
    max_j = 0
    for i in range(12,13):
        dataMatrix = convertToVector("E:\\derventData\\abstract_lda.txt",i,corpus_tfidf,dictionary,texts,titledata)
        N = 1000
        # centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]
        # data, y = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=[0.5, 0.25, 0.7, 0.5], random_state=0)
        # data = StandardScaler().fit_transform(data)

        # 主题降维，降成 维
        # pca=PCA(n_components=3)
        # dimention_reduced_newData = pca.fit_transform(dataMatrix)

        # matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
        # matplotlib.rcParams['axes.unicode_minus'] = False

        # plt.figure(figsize=(24, 16), facecolor='w')
        # plt.suptitle(u'DBSCAN聚类', fontsize=20)

        # fig = plt.figure(figsize=(24,24),facecolor='w')
        # ax=fig.add_subplot(111,projection='3d')

        # “数据1”的参数
        params = ((0.1, 5), (0.1, 10), (0.1, 15), (0.2, 5), (0.2, 10), (0.2, 15),(0.4, 15), (0.3, 5), (0.3, 10), (0.3, 15))

        # “数据2”的参数
        # params = ((0.5, 3), (0.5, 5), (0.5, 10), (1., 3), (1., 10), (1., 20))

        for j in range(len(params)):
            # print "hhhh"
            eps, min_samples = params[j]
            model = DBSCAN(eps=eps, min_samples=min_samples)
            # print dataMatrix
            model.fit(dataMatrix)
            y_hat = model.labels_
            if len(np.unique(y_hat))==1:
                continue
            core_indices = np.zeros_like(y_hat, dtype=bool)
            # print "jjjjj"
            # print np.unique(y_hat)
            pro_coeffi = metrics.silhouette_score(dataMatrix, model.labels_, metric="euclidean")

            print "主题数为%d的，参数为%d,轮廓系数为%f"%(i,j,pro_coeffi)
            if pro_coeffi>max_value:
                max_value = pro_coeffi
                max_i = i
                max_j = j
            # print y_hat
            # print core_indices
            # print model.core_sample_indices_
            core_indices[model.core_sample_indices_] = True

            y_unique = np.unique(y_hat)
            n_clusters = y_unique.size - (1 if -1 in y_hat else 0)
            print y_unique, '聚类簇的个数为：', n_clusters

        # clrs = []
        # for c in np.linspace(16711680, 255, y_unique.size):
        #     clrs.append('#%06x' % c)
        # plt.subplot(2, 3, i+1)
    print "最好的主题数为%d的，最好的参数为%d,最大的轮廓系数为%f"%(max_i,max_j,max_value)
        # position = 620+j+1
        # ax=fig.add_subplot(position,projection='3d')
        # print "i want to see"
        # print y_unique.size
        # print np.linspace(0, 0.8, y_unique.size)
        # clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size))
        # for k, clr in zip(y_unique, clrs):
        #     cur = (y_hat == k)
        #     print "current is "
        #     print cur
        #     if k == -1:
        #         ax.scatter(dimention_reduced_newData[cur, 0], dimention_reduced_newData[cur, 1], dimention_reduced_newData[cur, 2], s=20, c='k')
        #         continue
        #     ax.scatter(dimention_reduced_newData[cur, 0], dimention_reduced_newData[cur, 1], dimention_reduced_newData[cur, 2], s=30, c=clr, edgecolors='k')
        #     ax.scatter(dimention_reduced_newData[cur & core_indices][:, 0], dimention_reduced_newData[cur & core_indices][:, 1],dimention_reduced_newData[cur & core_indices][:, 2], s=60, c=clr, marker='o', edgecolors='k')
        # x1_min, x2_min = np.min(dimention_reduced_newData, axis=0)
        # x1_max, x2_max = np.max(dimention_reduced_newData, axis=0)
        # x1_min, x1_max = expand(x1_min, x1_max)
        # x2_min, x2_max = expand(x2_min, x2_max)
        # plt.xlim((x1_min, x1_max))
        # plt.ylim((x2_min, x2_max))
        # plt.grid(True)
        # ax.set_zlabel('Z') #坐标轴
        # ax.set_ylabel('Y')
        # ax.set_xlabel('X')
        # plt.title(ur'$\mu$ = %.1f  m = %d，聚类数目：%d' % (eps, min_samples, n_clusters), fontsize=16)
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.9)

    # plt.show()
# dimention_reduced_use_pca()
