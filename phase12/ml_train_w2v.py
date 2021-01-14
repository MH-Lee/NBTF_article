###############################################################################
##### 1. Base Package   #######################################################
###############################################################################
import json
import pandas as pd
import numpy as np
import re
import pickle
import os
import nltk
from time import time
from tqdm import tqdm
from ast import literal_eval
###############################################################################
##### 2. NLP Package   #######################################################
###############################################################################
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
###############################################################################
##### 3. ML Package   #######################################################
###############################################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from collections import namedtuple
from sklearn.model_selection import cross_val_score
import multiprocessing
import logging
import warnings

def make_w2v_data(X, word_vectorizer):
    i, j = 0, 0
    data = []
    for doc in tqdm(X):
        w2v_doc_mean = []
        for word in doc:
            try:
                i += 1
                w2v_doc_mean.append(word_vectorizer[word])
            except KeyError:
                j += 1
                w2v_doc_mean.append(np.random.normal(size=300))
        w2v_doc_mean = np.array(w2v_doc_mean).mean(axis=0)
        data.append(w2v_doc_mean)
    print(i, j)
    return np.array(data)

def return_two_calss_acc(clf, x_test, y_test):
    y_pred_prob = clf.predict_proba(x_test)
    L = np.argsort(-y_pred_prob, axis=1)
    two_pred = L[:,0:2]

    class_dic = {clf.classes_[i]: i for i in range(len(clf.classes_))}
    key_list = list(class_dic.keys())
    val_list = list(class_dic.values())

    dd = []
    score = []
    for i in range(len(y_test)):
        first = two_pred[i][0]
        second = two_pred[i][1]
        label = list([key_list[val_list.index(first)], key_list[val_list.index(second)]])
        if y_test[i] in label :
            score.append(1)
        else :
            score.append(0)
        dd.append({'y_test':y_test[i], 'first':label[0], 'second': label[1]})
    acc = sum(score)/len(y_test)
    return acc


warnings.filterwarnings("ignore")

w2v_model = './model/word_embedding/Word2vec1(base_token).model'
word_vector = Word2Vec.load(w2v_model)

data = pd.read_excel('./data/doc_set_final4.xlsx')
data['token'] = data.token.apply(lambda x: literal_eval(x))
X_data = data[['token', 'new_small_class']]
target_big = data.new_class.tolist()
target_small = data.new_small_class.tolist()

print("big class unique : ", len(np.unique(data.new_class)))
print("small class unique : ", len(np.unique(data.new_small_class)))

train_x, test_x, train_y, test_y = train_test_split(X_data, target_big,
                                                    test_size=0.3,
                                                    stratify=target_big,
                                                    shuffle=True,
                                                    random_state=1234)


train_y_small = train_x.new_small_class.tolist()
train_X = train_x.token.tolist()
test_y_small = test_x.new_small_class.tolist()
test_X = test_x.token.tolist()

X_train = make_w2v_data(train_X, word_vector)
X_test = make_w2v_data(test_X, word_vector)

big_class = np.unique(data.new_class)
small_class = np.unique(data.new_small_class)
big_class_dict  = {i: k for i, k in enumerate(big_class)}
small_class_dict  = {i: k for i, k in enumerate(small_class)}

class_dict = {
    key : list(np.unique(data.loc[data.new_class == key, 'new_small_class']))
    for key in big_class
}
idx_dict = {0:'big', 1:'small', 2:'class'}

### LogisticRegression
clf = LogisticRegression(solver='sag',  multi_class='multinomial', random_state=1234)

clf.fit(X_train, train_y)
filename = './model/ml_model/LR_clf_w2v.sav'
pickle.dump(clf, open(filename, 'wb'))

LR_scores = cross_val_score(clf, X_train, train_y, cv=10)
print("Logistics Regression CV Accuracy", LR_scores)
print("Logistics Regression CV 평균 Accuracy: ", np.mean(LR_scores))
print("Logistics Regression CV Std: ", np.std(LR_scores))

y_pred = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)
big_pred_prob = np.argsort(-y_pred_prob, axis=1)
print("Logistic Regression 1class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y)))
print("Logistic Regression 1class Recall: {:.4f}".format(recall_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class precision_score: {:.4f}".format(precision_score(test_y, y_pred, average='weighted')))
print("Logistic Regression 1class f1_score: {:.4f}".format(f1_score(test_y, y_pred, average='weighted')))
two_acc = return_two_calss_acc(clf, X_test, test_y)
print("Logistics Regression 2class 정확도: {:.3f}".format(two_acc))
two_pred = pd.DataFrame(big_pred_prob[:,0:2], columns=['predicted_label1', 'predicted_label2'], index=test_x.index)
print()
print()

len(np.unique(train_y_small))

clf.fit(X_train, train_y_small)
filename = './model/ml_model/LR_clf_small_w2v.sav'
pickle.dump(clf, open(filename, 'wb'))

LR_scores_small = cross_val_score(clf, X_train, train_y_small, cv=10)
print("Logistics Regression Small CV Accuracy", LR_scores_small)
print("Logistics Regression Small CV 평균 Accuracy: ", np.mean(LR_scores_small))
print("Logistics Regression Small CV Std: ", np.std(LR_scores_small))

y_pred = clf.predict(X_test)
y_pred_prob_small = clf.predict_proba(X_test)
small_predict_prob = np.argsort(-y_pred_prob_small, axis=1)
print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(y_pred, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, y_pred, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, y_pred, average='weighted')))

two_acc_small = return_two_calss_acc(clf, X_test, test_y_small)
print("Logistics Regression 2small class 정확도: {:.3f}".format(two_acc_small))


test_x['new_class'] = test_y
test_x = pd.concat([test_x, two_pred], axis=1)

test_x['predicted_label1'] = test_x.predicted_label1.map(big_class_dict)
test_x['predicted_label2'] = test_x.predicted_label2.map(big_class_dict)

small_class_dic = {clf.classes_[i]: i for i in range(len(clf.classes_))}
key_list = list(small_class_dic.keys())
val_list = list(small_class_dic.values())

score = []
predicted_small_label1 = []
predicted_small_label2 = []
for i in range(len(test_X)):
    sc_candidate = class_dict[test_x.iloc[i]['predicted_label1']] + class_dict[test_x.iloc[i]['predicted_label2']]
    sc_candidate = [x for x in sc_candidate if x in key_list]
    sc_candidate_idx = list({key: small_class_dic[key] for key in sc_candidate}.values())
    sc_candidate2 = [x for x in small_predict_prob[i] if x in sc_candidate_idx]
    first = sc_candidate2[0]
    second = sc_candidate2[1]
    label = list([key_list[val_list.index(first)], key_list[val_list.index(second)]])
    predicted_small_label1.append(key_list[val_list.index(first)])
    predicted_small_label2.append(key_list[val_list.index(second)])
    if test_y_small[i] in label :
        score.append(1)
    else :
        score.append(0)

acc = sum(score)/len(score)
print("hierachy accuracy_score", acc)

print("Logistic Regression 1small class 정확도: {:.4f}".format(accuracy_score(predicted_small_label1, test_y_small)))
print("Logistic Regression 1small class Recall: {:.4f}".format(recall_score(test_y_small, predicted_small_label1, average='weighted')))
print("Logistic Regression 1small class precision_score: {:.4f}".format(precision_score(test_y_small, predicted_small_label1, average='weighted')))
print("Logistic Regression 1small class f1_score: {:.4f}".format(f1_score(test_y_small, predicted_small_label1, average='weighted')))
