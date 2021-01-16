import pandas as pd
import os
import ast
import pickle
import json
import re
import numpy as np
from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import namedtuple
from nltk.probability import FreqDist
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from tqdm import tqdm
import itertools

with open('./data/label_dict/big.json', 'r') as f:
    big_class = json.load(f)
f.close()

with open('./data/label_dict/small.json', 'r') as f:
    small_class = json.load(f)
f.close()

with open('./data/label_dict/class.json', 'r') as f:
    class_dict = json.load(f)
f.close()

def text_cleaner(sents):
    text_data = sents.lower()
    change_list = [('모바일서비스', '모바일 서비스 '), ('\b클라우드펀드\b', '크라우드펀딩 '), ('웹툰', ' 웹툰 '), ('인터넷', '인터넷  '),
                   ('금융', ' 금융 '), ('간편송금', '간편 송금 '), ('k뷰티', 'k 뷰티 '), ('커머스', 'e커머스 '), ('컨텐츠', '콘텐츠 '),
                   ('모바일', '모바일 '), ('blockchain', '블록체인 '), ('게임', ' 게임 '), ('검색', ' 검색 '), ("p2p", " p2p "),
                   ('\bp2p\b', ' p2p '), ('\bhealth\b', '헬스 '),  ('\bhealthcare\b', ' 헬스케어 '), ('\b헬스케어\b', ' 헬스케어 '),
                   ('\b자산', ' 자산'), ('블록 체인', '블록체인 '), ('\b디지털헬스\b', '디지털 헬스케어 '), ('디지털헬스케어', '디지털 헬스케어 '),
                   ('machinelearning', '머신러닝 '), ('finance', '금융 '), ('finance', '금융 '), ('apps', '모바일앱'), ('어플리케이션', '모바일앱'),
                   ('의료', '의료 '), ('tutor', '교사 '),('이스포츠', 'e스포츠 '), ('e스포츠', 'e스포츠 '), ('콘텐츠', '콘텐츠 '), ('fintech', '핀테크 '),
                   ('라디오', ' 라디오 '), ('온오프라', '온오프라인'), ('정보보안', '정보보안 '), ('영상', '영상 '), ('네트워크', '네트워크 '), ('검사키트', ' 검사키트 '),
                   ('바이오', ' 바이오 '), ('bio', ' bio '), ('인공지능알고리즘', '인공지능 알고리즘'), ('o2o', 'o2o  '), ('교육', ' 교육  '), ('학습', ' 학습  '),
                   ('환자', ' 환자  '), ('로봇', ' 로봇  ')]

    for tu in change_list:
        text_data = re.sub(tu[0], tu[1], text_data)
    text_data = re.sub(r"\s{2,}", " ", text_data)
    return text_data

def stopwords_remove(corpus_data, sw_list):
    docs=[]
    for corpus_list in tqdm(corpus_data):
        words=[]
        for w in corpus_list:
            if w.isdecimal():
                continue
            if (w not in sw_list) & (len(w)>1):
                if w == '모바일게':
                    w = '모바일게임'
                words.append(w)
        docs.append(words)
    return docs

def noun_corpus(sents):
    noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
    noun_extractor.train(sents)
    nouns = noun_extractor.extract()

    noun_scores = {noun:score[0] for noun, score in nouns.items() if len(noun) > 1}
    tokenizer = NounLMatchTokenizer(noun_scores)
    corpus = [tokenizer.tokenize(sent) for sent in sents]
    return corpus

sw_list = pd.read_csv('./Preprocess/korean_stopwords.txt')['stopwords'].tolist()
com_list = pd.read_csv('./data/company_sw2.txt')['word'].tolist()
ext_list = pd.read_csv('./data/sw_extra.txt', '\t')['word'].tolist()
sw_list = list(sw_list+com_list+ext_list)

d2v_small_model_name = './model/word_embedding/Doc2vec_new_small2_4.model'
doc_vectorizer_small = Doc2Vec.load(d2v_small_model_name)

company_data = pd.read_excel('./phase34/extra/extra_data.xlsx')
company_data['Documents'] = company_data.Documents.apply(lambda x:text_cleaner(x))
big_corpus=noun_corpus(company_data['Documents'])
company_data['token'] = stopwords_remove(big_corpus, sw_list)
company_doc  = [doc_vectorizer_small.infer_vector(doc) for doc in company_data['token']]
token_list = company_data['token']
freq_list = list(itertools.chain.from_iterable(token_list))
freq_token = FreqDist(freq_list)
freq_df = pd.DataFrame(dict(freq_token).items(), columns=['word', 'freq'])
freq_df.to_excel('./data/zifs_compnay_extra.xlsx', index=False, encoding='cp949')


filename_big = './model/ml_model/LR_clf_last6.sav'
LR_clf =  pickle.load(open(filename_big, 'rb'))
y_pred = LR_clf.predict(company_doc)
y_pred_prob = LR_clf.predict_proba(company_doc)
big_pred_prob = np.argsort(-y_pred_prob, axis=1)
two_pred = pd.DataFrame(big_pred_prob[:,0:2], columns=['predicted_label1', 'predicted_label2'], index=company_data.index)
#two_pred_prob = pd.DataFrame([y_pred_prob[i,two_pred.iloc[i, 0:2].values].tolist() for i in range(two_pred.shape[0])],\
#                             columns = ['predicted_prob1', 'predicted_prob2'], index=company_data.index)

company_data = pd.concat([company_data, two_pred], axis=1)
company_data['predicted_label1'] = company_data.predicted_label1.apply(str).map(big_class)
company_data['predicted_label2'] = company_data.predicted_label2.apply(str).map(big_class)

filename_small = './model/ml_model/LR_clf_small_last6.sav'
LR_clf_small = pickle.load(open(filename_small, 'rb'))
# y_pred_small = LR_clf_small.predict(company_doc)
y_pred_prob_small = LR_clf_small.predict_proba(company_doc)
small_predict_prob = np.argsort(-y_pred_prob_small, axis=1)
small_class_dic = {LR_clf_small.classes_[i]: i for i in range(len(LR_clf_small.classes_))}
key_list = list(small_class_dic.keys())
val_list = list(small_class_dic.values())
company_data['predicted_small_label1'] = None
company_data['predicted_small_label2'] = None
company_data['predicted_small_label3'] = None
for i in range(len(company_data)):
    sc_candidate = class_dict[company_data.iloc[i]['predicted_label1']] + class_dict[company_data.iloc[i]['predicted_label2']]
    sc_candidate = [x for x in sc_candidate if x in key_list]
    sc_candidate_idx = list({key: small_class_dic[key] for key in sc_candidate}.values())
    sc_candidate2 = [x for x in small_predict_prob[i] if x in sc_candidate_idx]
    sc_candidate2_prob = y_pred_prob_small[i, sc_candidate2[:3]]
    first, first_prob = sc_candidate2[0], sc_candidate2_prob[0]
    second, second_prob = sc_candidate2[1], sc_candidate2_prob[1]
    third, third_prob = sc_candidate2[2], sc_candidate2_prob[2]
    company_data.loc[i,'predicted_small_label1'] = first
    company_data.loc[i,'predicted_small_label2'] = second
    company_data.loc[i,'predicted_small_label3'] = third

    # label = list([key_list[val_list.index(first)], key_list[val_list.index(second)]])
    # predicted_small_label1.append(key_list[val_list.index(first)])
    # predicted_small_label2.append(key_list[val_list.index(second)])

company_data['predicted_small_label1'] = company_data.predicted_small_label1.apply(str).map(small_class)
company_data['predicted_small_label2'] = company_data.predicted_small_label2.apply(str).map(small_class)
company_data['predicted_small_label3'] = company_data.predicted_small_label3.apply(str).map(small_class)
company_data.columns

company_data['token_len'] = company_data.token.apply(lambda  x : len(x))
company_data.to_excel('./phase34/extra/extra_results_3.xlsx', index=False, encoding='cp949')



'''
text_data = re.sub("[^가-힣\&]", " ", sents)
text_data = re.sub('모바일서비스', '모바일 서비스 ', text_data)
text_data = re.sub('\b클라우드펀드\b', '크라우드펀딩 ', text_data)
text_data = re.sub('웹툰', ' 웹툰 ', text_data)
text_data = re.sub('인터넷', '인터넷  ', text_data)
text_data = re.sub('금융', ' 금융 ', text_data)
text_data = re.sub('간편송금', '간편 송금 ', text_data)
text_data = re.sub('k뷰티', 'k 뷰티 ', text_data)
text_data = re.sub('커머스', 'e커머스 ', text_data)
text_data = re.sub('컨텐츠', '콘텐츠 ', text_data)
text_data = re.sub('모바일', '모바일 ', text_data)
text_data = re.sub('blockchain', '블록체인 ', text_data)
text_data = re.sub('검색', ' 검색 ', text_data)
text_data = re.sub("p2p", " p2p ", text_data)
text_data = re.sub("\bp2p\b", " p2p ", text_data)
text_data = re.sub('\bhealth\b', '헬스 ', text_data)
text_data = re.sub('\bhealthcare\b', ' 헬스케어 ', text_data)
text_data = re.sub('\b헬스케어\b', ' 헬스케어 ', text_data)
text_data = re.sub('\b자산', ' 자산', text_data)
text_data = re.sub('블록 체인', '블록체인 ', text_data)
text_data = re.sub('\b디지털헬스\b', '디지털 헬스케어 ', text_data)
text_data = re.sub('디지털헬스케어', '디지털 헬스케어 ', text_data)
text_data = re.sub('machinelearning', '머신러닝 ', text_data)
text_data = re.sub('finance', '금융 ', text_data)
text_data = re.sub('apps', '모바일앱', text_data)
text_data = re.sub('어플리케이션', '모바일앱', text_data)
text_data = re.sub('의료', '의료 ', text_data)
text_data = re.sub('tutor', '교사 ', text_data)
text_data = re.sub('이스포츠', 'e스포츠 ', text_data)
text_data = re.sub('e스포츠', 'e스포츠 ', text_data)
text_data = re.sub('콘텐츠', '콘텐츠 ', text_data)
text_data = re.sub('fintech', '핀테크 ', text_data)
text_data = re.sub('라디오', ' 라디오 ', text_data)
text_data = re.sub('온오프라', '온오프라인', text_data)
text_data = re.sub('정보보안', '정보보안 ', text_data)
text_data = re.sub('영상', '영상 ', text_data)
text_data = re.sub('네트워크', '네트워크 ', text_data)
text_data = re.sub('검사키트', ' 검사키트 ', text_data)
text_data = re.sub('바이오', ' 바이오 ', text_data)
text_data = re.sub('bio', ' bio ', text_data)
text_data = re.sub('인공지능알고리즘', '인공지능 알고리즘', text_data)
text_data = re.sub('o2o', 'o2o  ', text_data)
text_data = re.sub('교육', ' 교육  ', text_data)
text_data = re.sub('학습', ' 학습  ', text_data)
text_data = re.sub('환자', ' 환자  ', text_data)
text_data = re.sub('로봇', ' 로봇  ', text_data)
'''
