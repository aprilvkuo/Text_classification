#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: Main.py
@time: 2017/11/26 下午1:40
"""
import DataProcess,gensim
from gensim import corpora
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


def evaluate(x, y, class_name=LogisticRegression):
    clf = class_name(class_weight='balanced')
    print cross_validate(clf, x, y)


if __name__ == '__main__':
    y, class_map = DataProcess.class_counting()
    corpus = corpora.MmCorpus('./model/corpus_tf_idf_vector.mm')
    numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=corpus.num_terms).transpose()
    X = numpy_matrix
    print 'done'
    evaluate(X, y)

