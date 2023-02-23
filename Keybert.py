from keybert import KeyBERT
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
import pandas


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


data = pd.read_csv('data.csv')
lenth = data.shape[0]
model = SentenceTransformer('all-MiniLM-L12-v2')
# nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
nlp = spacy.load("en_core_web_lg", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

kw_model = KeyBERT(model=model)
f1 = 0
pre = 0
rec = 0
f1_l = []
pr_l = []
re_l = []
for j in range(12, 90, 2):
    print('{} start'.format(j))
    for i in range(lenth):
        keywords = kw_model.extract_keywords(data.iloc[i, 0], top_n=j)
        kw = [x[0] for x in keywords]
        gold = set(data['key words'][i].split(' '))
        f1s, p, r = f1_score(gold, kw)
        f1 += f1s
        pre += p
        rec += r
    print('{} finished'.format(j))
    f1_l.append(f1 / lenth)
    pr_l.append(pre / lenth)
    re_l.append(rec / lenth)
    f1 = 0
    pre = 0
    rec = 0

index = [x for x in range(12, 90, 2)]
df = pandas.DataFrame({'index': index, 'f1': f1_l, 'pr': pr_l, 're': re_l})
df.to_csv('keybert_res.csv')
