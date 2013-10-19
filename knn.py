# coding: utf-8
'''
用knn跑一遍，看看效果
'''

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import BallTree as BallTree
from sklearn.decomposition import TruncatedSVD

def read_csv():
    f = open("train.csv","U")
    reader = csv.reader(f)
    
    #存储语料
    train = []
    label = []

    a = 0
    
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            train.append(row[1])
            label.append(row[4:])
            
    f.close()

    f = open("test.csv","U")
    reader = csv.reader(f)
    
    #存储语料
    test = []
    ans = []

    a = 0
    
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            ans.append(row[0])
            test.append(row[1])
            
    f.close()
    
    return train,label,test,ans

if __name__ == "__main__":
    print "读文件"
    train,label,test,ans = read_csv()

    dim = 40
    
    vectorizer = TfidfVectorizer(max_features=None,min_df=4,max_df=0.4,sublinear_tf=False,ngram_range=(1,1),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode')

    length_train = len(train)
    x_all = train + test
    print "转化成tf-idf矩阵"
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    print x.shape
    print x

    print "svd"
    lsa = TruncatedSVD(n_components = dim )
    x = lsa.fit_transform(x)
    t = lsa.transform(t)
    print x
    print t.shape
    
    print "开始建树"
    bt = BallTree(x,leaf_size=1)

    print "开始查询"
    dist , ind = bt.query(t,k=1)

    print "写入文件"

    print ind
    
    head = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"

    f = open("answer_knn.csv","w")
    f.write(head+"\n")
    answer = [label[i[0]] for i in ind]

    for i in xrange(len(test)):
        to_string = ','.join(answer[i])
        f.write("%s,%s\n"%(ans[i],to_string))
    
    
    
    
    
