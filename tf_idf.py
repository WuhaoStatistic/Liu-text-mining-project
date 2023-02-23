import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as tfi
import collections
import pandas

data = pd.read_csv('data.csv')
nlp = spacy.load('en_core_web_lg')
lenn = data.shape[0]


def preprocess(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if (not token.is_stop and token.lemma_.isalpha())]


def keywords(text, n=10):
    idf_vec = vec.transform([text]).toarray()
    idf_df = pd.DataFrame({"tf_idf": idf_vec.flatten(), "word": vec.get_feature_names_out()})
    return idf_df.sort_values(by=['tf_idf', "word"], ascending=False)["word"].tolist()[:n]


# both these two should be python list
def f1_score(gold, pre):
    n = len(gold)
    tp = 0
    fp = 0
    for i in pre:
        if i in gold:
            tp += 1
        else:
            fp += 1
    fn = n - tp
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp), tp / (tp + fn)  # f1, pre, recall


vec = tfi(tokenizer=preprocess)
X = vec.fit_transform(data['text'])

f_l = []
f1 = 0
pre_l = []
pre = 0
rec_l = []
rec = 0
for j in range(12, 90, 2):
    for i in range(lenn):
        keyword_list = data['key words'][i].split(' ')
        pred = keywords(data['text'][i], n=j)
        f1s, pr, recal = f1_score(keyword_list, pred)
        f1 += f1s
        rec += recal
        pre += pr
    f_l.append(f1 / lenn)
    pre_l.append(pre / lenn)
    rec_l.append(rec / lenn)
    f1 = 0
    pre = 0
    rec = 0
    if j % 8 == 0:
        print('{} finished'.format(j))
index = [x for x in range(12, 90, 2)]
df = pandas.DataFrame({'index': index, 'f1': f_l, 'pr': pre_l, 're': rec_l})
df.to_csv('tf_res.csv')
