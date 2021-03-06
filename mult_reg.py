# coding: utf-8
'''
用线性回归跑一遍，然后归一化看看效果= =
'''

import csv
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import cross_validation


def ct(sent,sym):
    if sent.count(sym)!=0:
        return 1
    else:
        return 0

def find_rules(sent):
    """
    """
    tcount = []
    
    tcount.append(ct(sent,"!"))
    tcount.append(ct(sent,":)"))
    tcount.append(ct(sent,":("))
    tcount.append(ct(sent,"#"))
    tcount.append(ct(sent,"was"))
    tcount.append(ct(sent,"!"))
    tcount.append(ct(sent,":-]"))
    tcount.append(ct(sent,"%"))
    tcount.append(ct(sent,"=_="))
    tcount.append(ct(sent,"(:"))
    tcount.append(ct(sent,"?"))
    tcount.append(ct(sent,":D"))
    tcount.append(ct(sent,"tommoro"))
    tcount.append(1.0*len(sent)/100)
    tcount.append(ct(sent,":"))
    tcount.append(ct(sent,"{link}"))
    tcount.append(ct(sent,";)"))
    tcount.append(ct(sent,"="))
    tcount.append(ct(sent,":-P"))
    return tcount

    

def read_csv():
    f = open("train.csv","U")
    reader = csv.reader(f)
    
    train,label = [],[]

    a = 0
    etrain = []
    etest = []
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            train.append(row[1]+" "+row[2]+" "+row[3])
            sub_row = row[4:]
            sub_row = [float(i) for i in sub_row]
            label.append(sub_row)
            etrain.append(find_rules(row[1]))
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
            test.append(row[1]+" "+row[2]+" "+row[3])
            etest.append(find_rules(row[1]))
    f.close()

    return train,label,test,ans,etrain,etest

def remain(a,n):

    mn = 0.001
    for i in range(len(a)) :
        if a[i] < mn:
            a[i] = 0.0

    #for i in range(n):
    #    a[ind[i]] = 0

    mn = 1
    for i in range(len(a)) :
        if a[i] >= mn:
            a[i] = 1

    #a = a + abs(a.min())
    #a = 1.0*a/np.sum(a)
    if np.sum(a) < 1:
        a = 1.0*a/np.sum(a)
    
    return a 

def remain2(a,n):
    
    ind = range(len(a))
    ind.sort(lambda x,y:cmp(a[x],a[y]))

    mn = 0.001
    for i in range(len(a)) :
        if a[i] < mn:
            a[i] = 0.0

    mn = 1
    for i in range(len(a)) :
        if a[i] >= mn:
            a[i] = 1

    #a = a + abs(a.min())
    if np.sum(a) < 1 :
        a = 1.0*a/np.sum(a)

    return a

def remain3(a,n):
    for i in range(len(a)):
        if a[i]<0.1:
            a[i] = 0
        elif a[i] >0.9:
            a[i] = 1

    if np.sum(a) < 1:
        a = a/np.sum(a)

    return a

def readv():
    """
    """
    f = open("corpus/sent.txt")
    vocab = {}
    
    a = 0
    for line in f.readlines():
        if vocab.has_key(line.strip()):
            pass
        else:
            vocab[line.strip()] = a
            a += 1
    return vocab
    
if __name__ == "__main__":

    print "读文件"
    train,label,test,ans,etrain,etest = read_csv()

    print "读情感词表"
    vocab = readv()
    

    vectorizer = TfidfVectorizer(max_features=None,min_df=10,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode',use_idf=True,binary=False)

    sent_vectorizer = TfidfVectorizer(max_features=None,min_df=1,max_df=1.0,sublinear_tf=True,ngram_range=(1,1),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode',use_idf=False,binary=True,vocabulary=vocab)

    length_train = len(train)
    x_all = train + test
    print "转化成tf-idf矩阵"
    x_all = vectorizer.fit_transform(x_all)

    x = x_all[:length_train]
    t = x_all[length_train:]

    print "转化情感词表"
    sent_x_all = train + test
    sent_x_all = sent_vectorizer.fit_transform(sent_x_all)
    sent_x  = sent_x_all[:length_train]
    sent_t = sent_x_all[length_train:]

    print "合并"

    x = sparse.hstack((x,sent_x)).tocsr()
    t = sparse.hstack((t,sent_t)).tocsr()

    label = np.array(label)

    length_test = len(test)

    n = label.shape
    print "label shape",label.shape

    print "x shape",x.shape
    print "t shape",t.shape

    #构造结果的矩阵
    
    clf = linear_model.Ridge(alpha=2,fit_intercept=True,normalize=True,tol=1e-9,solver='auto')
    clf1 = linear_model.Ridge(alpha=0.001,fit_intercept=True,normalize=True,tol=1e-9,solver='auto')
    clf2 = linear_model.Ridge(alpha=2,fit_intercept=True,normalize=True,tol=1e-9,solver='auto')
    
    s = label[:,0:5]
    w = label[:,5:9]
    k = label[:,9:24]
    
    print "开始回归"

    clf.fit(x,label)
    train_x = clf.predict(x)
    train_t = clf.predict(t)
    print np.mean(cross_validation.cross_val_score(clf,x,label,cv=2,scoring='mean_squared_error',n_jobs=2))

    #print "开始训练情感词表"
    #clf.fit(sent_x,label)
    #train_x1 = clf.predict(sent_x)
    #train_t1 = clf.predict(sent_t)
    #print np.mean(cross_validation.cross_val_score(clf,sent_x,label,cv=2,scoring='mean_squared_error',n_jobs=2))

    #print "type",type(train_x1)

    #print "开始融合"
    train_x = np.hstack((train_x,etrain))
    train_t = np.hstack((train_t,etest))

    print "开始二次回归"

    clf1.fit(train_x,label)
    answer = clf1.predict(train_t)
    print np.mean(cross_validation.cross_val_score(clf1,train_x,label,cv=2,scoring='mean_squared_error',n_jobs=2))

    s = answer[:,0:5]
    w = answer[:,5:9]
    k = answer[:,9:24]
    
    
    head = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"
    
    f = open("ans_regression1.csv","w")
    f.write(head+"\n")

    for i in xrange(len(test)):
        ts,tw,tk = s[i],w[i],k[i]

        ts = remain(ts,0)
        tw = remain2(tw,0)
        tk = remain3(tk,13)

        str_s = [str(j) for j in ts]
        str_w = [str(j) for j in tw]
        str_k = [str(j) for j in tk]
        
        str_s = ','.join(str_s)
        str_w = ','.join(str_w)
        str_k = ','.join(str_k)
        
        f.write("%s,%s,%s,%s\n"%(ans[i],str_s,str_w,str_k))

    
    

