#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: ‘aprilkuo‘
@license: Apache Licence 
@contact: aprilvkuo@gmail.com
@site: 
@software: PyCharm Community Edition
@file: DataProcess.py
@time: 2017/11/26 上午12:13
"""
import jieba,os,gensim
from six import iteritems
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,CountVectorizer
import numpy as np
import pandas as pd
import pickle
from gensim import corpora, models
from sklearn.feature_selection import chi2,SelectKBest


class Corpus(object):
    """
     语料库的generator
    """

    def __init__(self,dir_name):
        self.dir_name = dir_name

    def __iter__(self):
        for file_name in os.listdir(self.dir_name):
            if os.path.isdir(os.path.join(self.dir_name, file_name)):
                files = map(lambda x: os.path.join(self.dir_name, file_name, x),\
                            os.listdir(os.path.join(self.dir_name, file_name)))
                for item in files:
                    doc = ''.join(map(lambda x:x.strip(), open(item, 'r').readlines()))
                    yield segment(doc)


def segment(line):
    """
    结巴分词调用
    :param line: 
    :return: 
    """
    return list(jieba.cut(line))


def build_dic(address='./data/answer'):
    """
    建立字典并保存
    :return: 
    """
    dictionary = corpora.Dictionary(line for line in Corpus(address))
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save('./model/dict.model')
    return dictionary


def doc2bow_vectoring(dictionary=None):
    """
    在字典上建立vsm
    :param dictionary: 
    :return: 
    """
    if not dictionary:
        dictionary = corpora.Dictionary.load('./model/dict.model')
    corpus = [dictionary.doc2bow(text) for text in Corpus('./data/answer')]
    corpora.MmCorpus.serialize('./model/corpora_doc2bow.mm', corpus)
    return corpora


def train_tf_idf(corpus=None):
    """
    在语料库上训练tf-idf模型
    :param corpus: 
    :return: 
    """
    if corpus is None:
        corpus = corpora.MmCorpus('./model/corpora_doc2bow.mm')
    tf_idf = models.TfidfModel(corpus)
    tf_idf.save('./model/tf_idf.model')
    return tf_idf


def corpus_vectoring(corpus=None, method='tf_idf', model=None):
    """
    将语料库向量化，这里可以用tf—idf，也可以用lda等
    :param corpus: 
    :param method: 
    :param model: 
    :return: 
    """
    if corpus is None:
        corpus = corpora.MmCorpus('./model/corpora_doc2bow.mm')
    if model is None:
        if method == 'tf_idf':
            model = models.TfidfModel.load('./model/tf_idf.model')
        else:
            model = None
    corpus_tf_idf = model[corpus]
    corpora.MmCorpus.serialize('./model/corpus_'+method+'_vector.mm', corpus_tf_idf)
    return corpus


def class_counting(dir_name='./data/answer'):
    #class_cnt = pd.DataFrame(columns=('class', 'cnt'))
    y = np.array([])
    class_map = dict()
    for file_name in os.listdir(dir_name):
        if os.path.isdir(os.path.join(dir_name, file_name)):
            files = map(lambda x: os.path.join(dir_name, file_name, x)\
                        , os.listdir(os.path.join(dir_name, file_name)))
            y = np.concatenate((y, len(class_map)*np.ones(len(files))))
            class_map[len(class_map)] = file_name
    #         class_cnt.loc[len(class_cnt)] = [file_name, len(files)]
    # pickle.dump(class_cnt, open('./model/class_cnt', 'w'))
    # print y
    # print class_map
    # return class_cnt
    return y, class_map

def feature_select():
    """
    特征选择，现在默认用filter模型选择卡方检验前5000个特征
    :return: 
    """
    y, class_map = class_counting()
    corpus = corpora.MmCorpus('./model/corpus_tf_idf_vector.mm')
    numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=corpus.num_terms).transpose()
    X = numpy_matrix
    X = SelectKBest(chi2, k=5000).fit_transform(X, y)
    pickle.dump(X, open("./model/feature_selected.mm", 'w'))




    #  build_dic()
    # test = Corpus('./data/answer')
    # for item in test:
    #     print '\n'.join(item)
    #     break
    #  test_0
    #  get_filenames('./data/answer')

    #  test_1
    # class_cnt = pd.DataFrame()
    # results = pickle.load(open('loaded_files.done','r'))
    # docs, class_cnt = results[0], results[1]
    # print len(docs), class_cnt.shape, class_cnt.head(10)
    # print class_cnt['cnt'].sum()
    # print '\n'.join(map(lambda x:x.decode('gbk',errors='ignore'),docs[:10]))



#
# def process_data(dir_name, save_name='tf-idf.model'):
#     vectoring = TfidfVectorizer(input='content', tokenizer=segment, analyzer='word',encoding='gbk')
#     content = []
#
#
#
#
#
#
#     for file_name in file_list:
#         before_size = len(content)
#         content.extend(open(file_name,'r').readlines())
#         file_cnt.append(len(content)-before_size)
#     x = vectoring.fit_transform(content)
#     y = np.concatenate((np.repeat([1], file_cnt[0],axis=0),
#                         np.repeat([0], file_cnt[1], axis=0)), axis=0)
#     data = pickle.dumps((x, y, vectoring))
#
#     with open(save_name, 'w') as f:
#         f.write(data)
#     return
#
#
# def load_data(model_name='tf-idf.model'):
#     '''
#     加载tf-idf数据
#     :param model_name:
#     :return:
#     '''
#     x, y, vectoring = pickle.loads(open(model_name,'r').read())
#     return x, y, vectoring