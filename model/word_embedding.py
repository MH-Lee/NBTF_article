import logging
import gensim
import pandas as pd
import multiprocessing
from collections import namedtuple
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from glove import Glove

def word2vec_model(text_data, **kwargs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(text_data,
                     window=kwargs['window'],
                     size=kwargs['num_features'],
                     min_alpha=kwargs['min_alpha'], # min learning-rate
                     min_count=kwargs['min_word_count'],
                     workers=kwargs['num_workers'],
                     seed=kwargs['seed'],
                     iter=kwargs['iter'],
                     sg=1,
                     hs=1,
                     negative = 10)
    return model

def doc2vec_model(text_data, **kwargs):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Doc2Vec(size=kwargs['size'],
                                  window=kwargs['window'],
                                  min_count=kwargs['min_count'],
                                  alpha=kwargs['alpha'],
                                  min_alpha=kwargs['min_alpha'],
                                  workers=kwargs['workers'],
                                  seed=kwargs['seed'],
                                  iter=kwargs['iter'])
    model.build_vocab(text_data)
    model.train(text_data,
                epochs=model.iter,
                total_examples=model.corpus_count)
    return model

def fasttext_model(text_data, **kwargs):
    model = FastText(text_data,
                     size=kwargs['size'],
                     window=kwargs['window'],
                     min_count=kwargs['min_count'],
                     workers=kwargs['workers'],
                     seed=kwargs['seed'],
                     sg=1)
    return model
