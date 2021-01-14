import pandas as pd
from ast import  literal_eval
from tqdm import tqdm
import nltk
import matplotlib.pyplot as plt

data = pd.read_excel('./data/doc_set_final5.xlsx')
data.token = data.token.apply(literal_eval)
# zipfs_law_sw =  pd.read_csv('./Preprocess/zipfs_law_sw3.txt', delimiter='\t')['word'].tolist()
# sw_list = list(set(zipfs_law_sw))

zipfs_law_sw =  pd.read_csv('./Preprocess/zipfs_law_small.txt', delimiter='\t')['word'].tolist()
sw_list = list(set(zipfs_law_sw))

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

data['token'] = stopwords_remove(data['token'], sw_list)
data_tokens = [ t for d in data['token'] for t in d]
data_text = nltk.Text(data_tokens, name='NMSC')
data_fdist = data_text.vocab()

data_fdist = pd.DataFrame.from_dict(data_fdist, orient='index')
data_fdist.columns = ['frequency']
data_fdist['term'] = list(data_fdist.index)
data_fdist = data_fdist.reset_index(drop=True)
data_fdist = data_fdist.sort_values(["frequency"], ascending=[False])
data_fdist = data_fdist.reset_index(drop=True)
data_fdist.to_excel('./frequency/zipf_law{}.xlsx'.format(6), index=False)

data_fdist['frequency'].plot()
plt.savefig('zipf_law.png')
plt.show()

data_fdist['frequency'].plot()
plt.xscale('log')
plt.yscale('log')
plt.savefig('zipf_law_log.png')
plt.show()

data.to_excel("./data/doc_set_final{}.xlsx".format(6), index=False, encoding='cp949')
