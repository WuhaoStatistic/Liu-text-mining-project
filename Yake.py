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
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp), tp / (tp + fn)  # f1, pre, recall


data = pd.read_csv('data.csv')
lenn = data.shape[0]

lan = 'en'
max_size = 1
ded_al = 'seqm'
w = 1
nk = range(12, 90, 2)

f1 = 0
pr = 0
rec = 0
f_l = []
pre_l = []
rec_l = []
print('start')
for th in [0.9]:
    for k in nk:
        model = yake.KeywordExtractor(n=max_size, dedupLim=th, dedupFunc=ded_al, windowsSize=w, top=k)
        for i in range(lenn):
            keywords = model.extract_keywords(data.iloc[i, 0])
            key = [x[0] for x in keywords]
            gld = keyword_list = data['key words'][i].split(' ')
            f1s, ps, rs = f1_score(gld, key)
            f1 += f1s
            pr += ps
            rec += rs
        f_l.append(f1/lenn)
        pre_l.append(pr/lenn)
        rec_l.append(rec/lenn)
        f1 = 0
        pr = 0
        rec = 0
        print('{} finished'.format(k))
index = [x for x in range(12, 90, 2)]
df = pd.DataFrame({'index': index, 'f1': f_l, 'pr': pre_l, 're': rec_l})
df.to_csv('yake_res.csv')
