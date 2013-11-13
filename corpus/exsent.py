#coding: utf-8

import csv

if __name__ == '__main__':
    a = range(1,6)
    a = [str(i) for i in a]

    words = []

    for i in a:
        fi = i+".txt"
        f = open(fi)
        for line in f.readlines():
            if len(line.split()) == 1:
                words.append(line)
        f.close()

    f = open("sent.txt","w")

    print len(words)

    f = open("../train.csv")

    reader = csv.reader(f)

    corpus = " "

    print "read train test"

    for row in reader:
        tweet = row[1].lower()
        g = lambda x : x if x.isalpha() else " "
        tweet = filter(g,tweet)
        corpus += tweet
    f.close()    

    f = open("../test.csv")

    reader = csv.reader(f)

    for row in reader:
        tweet = row[1].lower()
        g = lambda x : x if x.isalpha() else " "
        tweet = filter(g,tweet)
        corpus += tweet

    print "make corpus a set"

    corpus = corpus.split()
    corpus = set(corpus)    

    print "len corpus",len(corpus)

    f.close()
    f = open("sent.txt","w")

    a = 0
    for word in words:
        if word.strip() in corpus:
            a += 1
            f.write(word)

    print a
