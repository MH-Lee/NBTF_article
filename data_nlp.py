import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import konlpy
from konlpy.tag import Kkma, Okt, Hannanum, Twitter
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from soynlp.tokenizer import LTokenizer
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import NounLMatchTokenizer
from tqdm import tqdm
from multiprocessing import Process



def text_cleaner(sents):
    text_data = re.sub("[^가-힣A-z&235]", " ", sents)
    text_data = re.sub("[\r.*?\n]", " ", text_data)
    text_data = re.sub("\[", " ", text_data)
    text_data = re.sub("\]", " ", text_data)
    text_data = re.sub("\x0c", " ", text_data)
    text_data = re.sub('크라우드펀딩$', ' 크라우드펀딩 ',text_data)
    text_data = text_data.lower()
    text_data = text_data.replace('-', ' ')
    text_data = text_data.replace("sw", " 소프트웨어 ")
    text_data = re.sub("3D프린팅", " 3D프린팅 ", text_data)
    text_data = re.sub("3D프린터", " 3D프린터 ", text_data)
    text_data = re.sub("p2p크라우드펀딩", " p2p 크라우드펀딩 ", text_data)
    text_data = re.sub("\bp2p\b", " p2p ", text_data)
    text_data = re.sub("device", " 디바이스 ", text_data)
    text_data = text_data.replace("2d3d", "2d 3d ")
    text_data = text_data.replace("games", "게임 ")
    text_data = text_data.replace("교육", "교육 ")
    text_data = text_data.replace("학습", "학 ")
    text_data = text_data.replace("인공지능", "인공지능 ")
    text_data = text_data.replace("5세대5g", " 5g ")
    text_data = text_data.replace("e스포츠", " e스포츠 ")
    text_data = text_data.replace("arvr", "vr ar ")
    text_data = text_data.replace("vrar", "vr ar")
    text_data = text_data.replace("가상현실vr증강현실ar", "vr ar")
    text_data = text_data.replace("가상현실증강현실", "vr ar")
    text_data = text_data.replace("가상현실vr", "vr")
    text_data = text_data.replace("가상현실vr", "vr")
    text_data = text_data.replace("vr가상현실", "vr")
    text_data = text_data.replace("증강현실ar", "ar")
    text_data = text_data.replace("비정형정형", "비정형 정형")
    for sys_ in ["system", "systems", "systems", "System", "Systems", "SYSTEM"]:
        text_data = text_data.replace(sys_, "")
    text_data = text_data.replace("Systems미국", "")
    text_data = re.sub(r"\s{2,}", " ", text_data)
    return text_data

def noun_corpus(sents):
    noun_extractor = LRNounExtractor_v2(verbose=True, extract_compound=True)
    noun_extractor.train(sents)
    nouns = noun_extractor.extract()

    noun_scores = {noun:score[0] for noun, score in nouns.items() if len(noun) > 1}
    tokenizer = NounLMatchTokenizer(noun_scores)
    corpus = [tokenizer.tokenize(sent) for sent in sents]
    return corpus

def stopwords_remove(corpus_data, sw_list):
    docs=[]
    for corpus_list in tqdm(corpus_data):
        words=[]
        for w in corpus_list:
            if w.isdecimal():
                continue
            if (w not in sw_list) & (len(w)>1):
                words.append(w)
        docs.append(words)
    return docs

if __name__ == '__main__':
    process = input("가공차수 입력")
    print(process)
    data = pd.read_excel("./data/doc_set_final{}.xlsx".format(process))
    stopwords_list = pd.read_csv('./Preprocess/korean_stopwords.txt')['stopwords'].tolist()
    if int(process) == 0:
        sw_list = list(set(stopwords_list))
    else:
        zipfs_law_sw =  pd.read_csv('./Preprocess/zipfs_law_sw3.txt', delimiter='\t')['word'].tolist()
        sw_list = list(set(stopwords_list+zipfs_law_sw))
    data['content'] = data.content.apply(lambda x:text_cleaner(x))
    big_corpus=noun_corpus(data['content'])
    data['token'] = stopwords_remove(big_corpus, sw_list)
    # data['token'] = proc
    data_tokens = [ t for d in data['token'] for t in d]
    data_text = nltk.Text(data_tokens, name='NMSC')
    data_fdist = data_text.vocab()

    data_fdist = pd.DataFrame.from_dict(data_fdist, orient='index')
    data_fdist.columns = ['frequency']
    data_fdist['term'] = list(data_fdist.index)
    data_fdist = data_fdist.reset_index(drop=True)
    data_fdist = data_fdist.sort_values(["frequency"], ascending=[False])
    data_fdist = data_fdist.reset_index(drop=True)
    data_fdist.to_excel('./frequency/zipf_law{}.xlsx'.format(int(process) + 1), index=False)

    data_fdist['frequency'].plot()
    plt.savefig('./frequency/zipf_law.png')
    plt.show()

    data_fdist['frequency'].plot()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('./frequency/zipf_law_log.png')
    plt.show()

    data.to_excel("./data/doc_set_final{}.xlsx".format(int(process) + 1), index=False, encoding='cp949')
