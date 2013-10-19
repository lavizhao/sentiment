#coding: utf-8
'''
打算用lda试一下，对于情感分析，可能lda效果不好，不过可以用在文本分类上，可以先用knn试一下
'''

import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import BallTree as BallTree

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

    print "开始建树"
    bt = BallTree(train,leaf_size = 2)

    print "开始查询"

    dist,ind = bt.query(test,k=10)

    head = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"

    print ind.shape
    
    answer = []

    t = open("answer_lda.csv","w")
    t.write(head+'\n')

    for i in xrange(len(test)):
        temp = [0.0 for i in range(24)]
        for j in ind[i]:
            temp = temp+label[j]
        temp = temp/np.sum(temp)
        
        temp = [str(k) for k in temp]

        to_str = ','.join(temp)

        t.write("%s,%s\n"%(ans[i],to_str))
