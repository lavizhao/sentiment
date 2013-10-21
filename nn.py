#coding: utf-8
'''
打算用lda试一下，对于情感分析，可能lda效果不好，不过可以用在文本分类上，可以先用knn试一下
'''

import csv
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.feature_extraction.text import TfidfVectorizer

def read_csv():
    f = open("train.csv","U")
    reader = csv.reader(f)
    
    train,label = [],[]

    a = 0
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            train.append(row[1])
            sub_row = row[4:]
            sub_row = [float(i) for i in sub_row]
            label.append(sub_row)
    f.close()

    f = open("test.csv","U")
    reader = csv.reader(f)
    test,ans = [],[]

    a = 0
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            ans.append(int(row[0]))
            test.append(row[1])
    f.close()

    return train,label,test,ans

def remain(a,n):

    a = 1.0*a/np.sum(a)

    return a 


if __name__ == "__main__":
    print "读文件"
    train,label,test,ans = read_csv()

    vectorizer = TfidfVectorizer(max_features=None,min_df=30,max_df=1.0,sublinear_tf=True,ngram_range=(1,1),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode',use_idf=True,binary=False)

    length_train = len(train)
    x_all = train + test
    print "转化成tf-idf矩阵"
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    label = np.array(label)

    length_test = len(test)

    n = label.shape[1]

    print "x shape",x.shape
    print "t shape",t.shape

    x = x.toarray()
    t = t.toarray()



    ran = [[0.0,1.0] for i in range(100)]

    print "建立神经网络"

    #建立神经网络
    fnn = buildNetwork(x.shape[1],1000,20,24,bias=True)

    print "建立数据集"
    #建立数据集
    ds = SupervisedDataSet(x.shape[1], 24)
    for i in range(len(train)):
        ds.addSample(x[i],label[i])
    
    print "构造bp"
    #构造bp训练集
    trainer = BackpropTrainer( fnn, ds, momentum=0.1, verbose=True, weightdecay=0.01)
    print "开始训练"
    trainer.trainEpochs(epochs=100)

    print "开始返回结果"
    out = SupervisedDataSet(x.shape[1], 24)
    for i in range(len(test)):
        temp = [0 for j in range(24)]
        out.addSample(t[i],temp)

    out = fnn.activateOnDataset(out)

    s = out[:,0:5]
    w = out[:,5:9]
    k = out[:,9:24]

    print "write"
    head = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"
    t = open("new_nn.csv","w")
    t.write(head+"\n")
    
    
    for i in xrange(len(test)):
        ts,tw,tk = s[i],w[i],k[i]

        #ts = remain(ts,2)
        #tw = remain(tw,2)
        #tk = remain(tk,12)

        str_s = [str(j) for j in ts]
        str_w = [str(j) for j in tw]
        str_k = [str(j) for j in tk]
        
        str_s = ','.join(str_s)
        str_w = ','.join(str_w)
        str_k = ','.join(str_k)
        
        t.write("%s,%s,%s,%s\n"%(ans[i],str_s,str_w,str_k))


