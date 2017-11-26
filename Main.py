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
import numpy as np
import DataProcess, gensim
from gensim import corpora
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import ensemble
from sklearn import svm
from sklearn.feature_selection import chi2,SelectKBest
import pickle


def evaluate(x, y, class_name=LogisticRegression):
    print class_name.__name__
    clf = class_name(class_weight='balanced')
    print cross_validate(clf, x, y)


if __name__ == '__main__':
    models = [LogisticRegression, ensemble.RandomForestClassifier, svm.LinearSVC]
    y, class_map = DataProcess.class_counting()
    y = y.astype(np.int64)
    print y.dtype, type(y)
    print class_map
    print np.bincount(np.array(y))
    X = pickle.load(open('./model/feature_selected.mm', 'r'))
    for item in models:
        evaluate(X, y, item)

