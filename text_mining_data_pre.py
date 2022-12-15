import sys
import os
import spacy
import pandas as pd

path = "E:\\pycharm\\gpu_general\\Inspec"

docpath = path + "\\docsutf8"
keypath = path + "\\keys"

data = pd.DataFrame(columns=['text', 'key words'], index=[x for x in range(2000)])
count = 0
# for path, dir_list, file_list in os.walk(keypath):
#     for file_name in file_list:
#         prefix = file_name.split('.')[0]
#         os.rename(keypath + "\\" + file_name,keypath + "\\" + prefix+'.txt')

for path, dir_list, file_list in os.walk(keypath):
    for file_name in file_list:
        with open(keypath + "\\" + file_name, "rb") as f:
            data.iloc[count, 1] = f.read()
            count += 1

count = 0
for path, dir_list, file_list in os.walk(docpath):
    for file_name in file_list:
        with open(docpath + "\\" + file_name, "rb") as f:
            data.iloc[count, 0] = f.read()
            count += 1
data.to_csv('data.csv')
print(data.head())
