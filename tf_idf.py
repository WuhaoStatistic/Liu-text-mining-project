import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer as tfi
import collections

data = pd.read_csv('data.csv')
nlp = spacy.load('en_core_web_sm')
lenn = data.shape[0]


def preprocess(text):
    doc = nlp(text)
    return [token.lemma_ for token in doc if (not token.is_stop and token.lemma_.isalpha())]


def keywords(text, n=10):
    idf_vec = vec.transform([text]).toarray()
    idf_df = pd.DataFrame({"tf_idf": idf_vec.flatten(), "word": vec.get_feature_names()})
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
    return 2 * tp / (2 * tp + fp + fn)


vec = tfi(tokenizer=preprocess)
X = vec.fit_transform(data['text'])

f_l = []
f1 = 0
for j in [25, 26, 27, 28, 29]:
    for i in range(lenn):
        keyword_list = data['key words'][i].split(' ')
        pred = keywords(data['text'][i], n=j)
        f1 += f1_score(keyword_list, pred)
    f_l.append(f1 / lenn)
    f1 = 0
print(f_l)
# 0.2136
