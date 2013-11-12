# coding: utf-8
'''
用线性回归跑一遍，然后归一化看看效果= =
'''

import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn import cross_validation


def read_csv():
    f = open("train.csv","U")
    reader = csv.reader(f)
    
    train,label = [],[]

    a = 0
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            train.append(row[1]+" "+row[2]+" "+row[3])
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

    mn = 0.001
    for i in range(len(a)) :
        if a[i] < mn:
            a[i] = 0.0

    for i in range(n):
        a[ind[i]] = 0

    a = a + abs(a.min())
    a = 1.0*a/np.sum(a)
    
    a = a/np.sum(a)
    return a 

def remain2(a,n):
    
    ind = range(len(a))
    ind.sort(lambda x,y:cmp(a[x],a[y]))

    mn = 0.001
    for i in range(len(a)) :
        if a[i] < mn:
            a[i] = 0.0


    for i in range(n):
        a[ind[i]] = 0

    a = a + abs(a.min())
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

if __name__ == "__main__":

    print "读文件"
    train,label,test,ans = read_csv()

    vectorizer = TfidfVectorizer(max_features=None,min_df=10,max_df=1.0,sublinear_tf=True,ngram_range=(1,2),smooth_idf=True,token_pattern=r'\w{1,}',analyzer='word',strip_accents='unicode',use_idf=True,binary=False)

    length_train = len(train)
    x_all = train + test
    print "转化成tf-idf矩阵"
    x_all = vectorizer.fit_transform(x_all)
    x = x_all[:length_train]
    t = x_all[length_train:]

    label = np.array(label)

    length_test = len(test)

    n = label.shape
    print "label shape",label.shape

    print "x shape",x.shape
    print "t shape",t.shape
    print x[0]
    

    #构造结果的矩阵
    answer = []
    
    clf = linear_model.Ridge(alpha=2,fit_intercept=True,normalize=True,tol=1e-9,solver='auto')
    clf1 = linear_model.Ridge(alpha=2,fit_intercept=True,normalize=True,tol=1e-9,solver='auto')
    clf2 = linear_model.Ridge(alpha=2,fit_intercept=True,normalize=True,tol=1e-9,solver='auto')
    
    s = label[:,0:5]
    w = label[:,5:9]
    k = label[:,9:24]
    
    print "开始回归"

    print "do s"
    clf.fit(x,s)
    s_answer = clf.predict(t)
    print np.mean(cross_validation.cross_val_score(clf,x,s,cv=2,scoring='mean_squared_error',n_jobs=2))

    print "do w"
    clf1.fit(x,w)
    w_answer = clf1.predict(t)
    print np.mean(cross_validation.cross_val_score(clf1,x,w,cv=2,scoring='mean_squared_error',n_jobs=2))

    print "do k"
    clf2.fit(x,k)
    k_answer = clf2.predict(t)
    print np.mean(cross_validation.cross_val_score(clf2,x,k,cv=2,scoring='mean_squared_error',n_jobs=2))

    answer = np.array(answer)
    answer = answer.T
    
    print answer.shape

    print "写入文件"
    head = "id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15"
    
    f = open("ans_regression1.csv","w")
    f.write(head+"\n")

    for i in xrange(len(test)):
        ts,tw,tk = s_answer[i],w_answer[i],k_answer[i]

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

    
    

