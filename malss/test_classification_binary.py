# -*- coding: utf-8 -*-

from malss import MALSS
import pandas as pd

def __make_data():
    import os

    if not os.path.exists('data/spam.txt'):
        if not os.path.exists('data'):
            os.mkdir('data')

        import urllib
        urllib.urlretrieve(
            'http://mldata.org/repository/data/download/uci-20070111-spambase/',
            'data/uci-20070111-spambase.arff')
        fi = open('data/uci-20070111-spambase.arff', 'r')
        fo = open('data/spam.txt', 'w')
        header = []
        for line in fi:
            if '@attribute' in line:
                header.append(line.rstrip().split(' ')[1])
            if '@data' in line:
                break
        for h in header:
            fo.write(h)
            if h == header[-1]:
                fo.write('\n')
            else:
                fo.write(',')
        for line in fi:
            line = line.rstrip()
            line = line[:-1]+'spam' if line[-1] == '1' else line[:-1]+'nonspam'
            fo.write(line+'\n')
        fi.close()
        fo.close()

        os.remove('data/uci-20070111-spambase.arff')

    data = pd.read_csv('./data/spam.txt', header=0)
    y = data['class']
    del data['class']

    return data, y


if __name__ == "__main__":
    data, y = __make_data()
    cls = MALSS(data, y, 'classification', n_jobs=3)
    cls.execute()
