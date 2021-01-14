import os
import sys
sys.path.append(os.pardir)
import logging
import gensim
import pandas as pd
import multiprocessing
from ast import literal_eval
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts, get_tmpfile
from glove import Corpus, Glove
from sklearn.model_selection import train_test_split
from NTBF_TOA.model.word_embedding import *
# from model.word_embedding import word2vec_model, doc2vec_model, fasttext_model
import json



def main(model_select):
    data = pd.read_excel("./data/doc_set_final4.xlsx")
    data.token = data.token.apply(lambda x : literal_eval(x))
    data = data.sample(frac=1, random_state=1234)

    token_list = data.token.tolist()
    target = data[['new_class', 'new_small_class']]
    train_x_data, test_x_data, train_y, test_y = train_test_split(token_list, target,
                                                                    test_size=0.3,
                                                                    stratify=target,
                                                                    shuffle=True,
                                                                    random_state=1234)

    if model_select == 'w2v':
        w2v_name = 'base_token'
        print("모델 학습")
        word2vec_kargs = {'num_features':300,
                          'num_workers':4,
                          'window':8,
                          'seed':1234,
                          'min_word_count':5,
                          'min_alpha':0.025,
                          'iter':30}
        model = word2vec_model(train_x_data, **word2vec_kargs)
        print("모델 저장")
        model_name = './model/word_embedding/Word2vec1({}).model'.format(w2v_name)
        model.save(model_name)

    elif model_select == 'd2v':
        TaggedDocument = namedtuple('TaggedDocument', 'words tags')
        tagged_train_docs = [TaggedDocument(d, [c[1]['new_class'], c[1]['new_small_class']]) for d, c in zip(train_x_data, train_y.iterrows())]
        print("모델 학습")
        doc2vec_kargs = {'size':300,
                         'window':8,
                         'min_count':5,
                         'alpha':0.025,
                         'min_alpha':0.025,
                         'workers':4,
                         'seed':1234,
                         'iter':50}
        model = doc2vec_model(tagged_train_docs, **doc2vec_kargs)
        print("모델 저장")
        model.save('./model/word_embedding/Doc2vec_new_small2_4.model')

    elif model_select == 'fasttext':
        print("모델 학습")
        ft_kargs = {'size':300,
                    'window':5,
                    'min_count':3,
                    'workers':4,
                    'seed':1234}
        model = fasttext_model(train_x_data, **ft_kargs)
        print("모델 저장")
        model.save('./model/word_embedding/FastText.model')

    elif model_select == 'glove':
        glove_kargs = {'size':300, 'lr':0.005, 'random_state':1234, 'no_threads':4, 'epoch':30}
        corpus = Corpus()
        corpus.fit(train_x_data, window=8)
        glove = Glove(no_components=glove_kargs['size'], learning_rate=glove_kargs['lr'])
        glove.fit(corpus.matrix, epochs=glove_kargs['epoch'], no_threads=glove_kargs['no_threads'], verbose=True)
        glove.add_dictionary(corpus.dictionary)
        print("모델 저장")
        glove.save('./model/word_embedding/glove.model')
    else:
        print("3가지 방식 중에 고르시오")

if __name__ == '__main__':
    model_select = input("생성하려는 word embedding 모델을 고르시오 (1) w2v (2) d2v (3) fasttext (4) glove")
    main(model_select)
