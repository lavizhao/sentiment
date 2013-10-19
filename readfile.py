#coding: utf-8

import csv

def read_file():
    
    print "读文件"
    f = open("train.csv","U")
    reader = csv.reader(f)

    t = open("total.txt","w")

    train = []
    test = []
    
    a = 0
    print "读训练文件"
    for row in reader :
        if a == 0:
            a = a + 1
        else:
            train.append(1)
            t.write(row[2]+" "+row[0]+" "+row[3]+"\n")

    f.close()

    a = 0
    f = open("test.csv","U")
    reader = csv.reader(f)
    
    print "读测试文件"
    for row in reader:
        if a == 0:
            a = a + 1
        else:
            test.append(2)
            t.write(row[2]+" "+row[0]+" "+row[3]+"\n")

    print "train",len(train)
    print "test",len(test)

if __name__ == "__main__":
    read_file()
