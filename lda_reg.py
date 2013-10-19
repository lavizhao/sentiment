#coding: utf-8
'''
打算用lda试一下，对于情感分析，可能lda效果不好，不过可以用在文本分类上，可以先用knn试一下
'''

import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import BallTree as BallTree
from sklearn import linear_model

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
    ind = range(len(a))
    ind.sort(lambda x,y:cmp(a[x],a[y]))

    for i in range(n):
        a[ind[i]] = 0

    a = a + abs(a.min())
    a = 1.0*a/np.sum(a)

    return a 

if __name__ == "__main__":
    print "读文件"
    train,label,test,ans = read_csv()

    train,test = [],[]

    print "读topic"
    f1 = open("topic_train.txt")
    for line in f1.readlines():
        sp = line.split()
        sp = [float(j) for j in sp]

        train.append(sp)

    f2 = open("topic_test.txt")
    for line in f2.readlines():
        sp = line.split()
        sp = [float(j) for j in sp]

        test.append(sp)

    train = np.array(train)
    test  = np.array(test)

    label = np.array(label)

    x = train
    t = test

    answer = []

    n = label.shape[1]
    
    for i in range(n):
        print "第%s个"%(i)
        #clf = linear_model.LinearRegression()
        clf = linear_model.Ridge(alpha=2.5)
        #clf = linear_model.ElasticNet(alpha=100,l1_ratio=0.96)
        #clf = SVR(gamma = 0.1)
        clf.fit(x,label[:,i])
        temp_answer = clf.predict(t)
        answer.append(temp_answer)

    answer = np.array(answer)
    answer = answer.T
    
    print answer.shape

    print "归一化"
    s = answer[:,0:5]
    w = answer[:,5:9]
    k = answer[:,9:24]

    print "w shape",w.shape
    print "k shpae",k.shape

    #s = s/np.mean(s,axis=1)
    #w = w/np.mean(w,axis=1)
    #k = k/np.mean(k,axis=1)

    print "写入文件"
    head = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"
    
    f = open("ans_regression.csv","w")
    f.write(head+"\n")

    for i in xrange(len(test)):
        ts,tw,tk = s[i],w[i],k[i]

        #ts = ts/np.sum(ts)
        #tw = tw/np.sum(tw)
        #tk = tk/np.sum(tk)

        ts = remain(ts,2)
        tw = remain(tw,2)
        tk = remain(tk,12)

        str_s = [str(j) for j in ts]
        str_w = [str(j) for j in tw]
        str_k = [str(j) for j in tk]
        
        str_s = ','.join(str_s)
        str_w = ','.join(str_w)
        str_k = ','.join(str_k)
        
        f.write("%s,%s,%s,%s\n"%(ans[i],str_s,str_w,str_k))

    
