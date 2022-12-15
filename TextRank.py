
from summa import keywords, summarizer
import pandas as pd
#import pytextrank

# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe('textrank')


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


#
#nlp = spacy.load("en_core_web_sm")

#nlp.add_pipe("textrank")
data = pd.read_csv('data.csv')
lenth = len(data)
f1 = 0
for i in range(lenth):
    #doc = nlp()
    a = keywords.keywords(data.iloc[i, 0]).split('\n')
    b = [x.split(' ') for x in a]
    key = list(set([i for item in b for i in item]))
    gld = keyword_list = data['key words'][i].split(' ')
    f1 += f1_score(gld, key)
    if i in [10, 100, 200, 500, 1500, 1800]:
        print(i)
print(f1 / lenth)

# text = """Separate accounts go mainstream [investment] New entrants are shaking up the separate account industry by supplying Web based platforms that give advisers the tools to pick independent money managers '
# ."""
#
# a = keywords.keywords(text).split('\n')
# b = [x.split(' ') for x in a]
# print(list(set([i for item in b for i in item])))
