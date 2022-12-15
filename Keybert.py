from keybert import KeyBERT
import pandas as pd
import gensim
import gensim.downloader as api
import spacy


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


data = pd.read_csv('data.csv')
lenth = data.shape[0]

nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

kw_model = KeyBERT(model=nlp)
f1 = 0

for i in range(lenth):
    keywords = kw_model.extract_keywords(data.iloc[i, 0], use_mmr=True, diversity=0.15, top_n=28)
    kw = [x[0] for x in keywords]
    gold = set(data['key words'][i].split(' '))
    f1 += f1_score(gold, kw)
    if i in [10, 100, 500, 1000, 1800]:
        print(i)

print(f1 / lenth)
