import yake
import pandas as pd


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
lenn = data.shape[0]

lan = 'en'
max_size = 1
ded_th = [0.5, 0.6, 0.7, 0.8, 0.9]
ded_al = 'seqm'
w = 1
nk = [22, 23, 24, 25]

f1 = 0
print('start')
for th in ded_th:
    for k in nk:
        model = yake.KeywordExtractor(n=max_size, dedupLim=th, dedupFunc=ded_al, windowsSize=w, top=k)
        for i in range(lenn):
            keywords = model.extract_keywords(data.iloc[i, 0])
            key = [x[0] for x in keywords]
            gld = keyword_list = data['key words'][i].split(' ')
            f1 += f1_score(gld, key)
        print('para is th:{} nk:{},f1 :{}'.format(th, k, f1 / lenn))
        f1=0