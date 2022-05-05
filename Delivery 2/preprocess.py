import csv
import json
import os
from collections import Counter
from typing import re
import pandas as pd
from nltk.corpus import stopwords

stop_words = set(stopwords.words('turkish'))


def main():
    # defining the stopwords
    stop_words = stopwords.words('turkish')
    # path of the input json folder
    path = "2021-01-20220322T055600Z-001\\2021-01\\"
    # count the number of json file ( for different input folders)
    dir_path = r'2021-01-20220322T055600Z-001\\2021-01\\'
    num_of_file = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
    print(num_of_file)

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    language = "tr"
    max_ngram_size = 3
    deduplication_thresold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 1
    numOfKeywords = 10

    with open('labeled_suc.csv', 'w', newline='',errors='ignore') as file:
        writer = csv.writer(file)
        writer.writerow(["v1", "v2"])

    i = 1
    while i < num_of_file + 1:
        print(i)
        fpath = path + str(i) + ".json"

        with open(fpath,'r',encoding="utf-8",errors='ignore') as fh:
            data = json.load(fh)
        suc = str(data["Suç"])
        ictihat = str(data["ictihat"])
        suc = re.sub(r'[^\w\s]', '', suc)
        suc = " ".join(suc.split())
        suc = suc.lower()
        ictihat = re.sub(r'[^\w\s]', '', ictihat)
        ictihat = " ".join(ictihat.split())
        ictihat = ictihat.lower()
        if data["Suç"] != "" :
            with open('labeled_suc.csv', 'a', newline='',errors='ignore') as file:
                writer = csv.writer(file)
                writer.writerow([suc, ictihat])
        i = i+1
        fh.close()

    file_name = "labeled_suc.csv"
    df = pd.read_csv(file_name, encoding="ISO-8859-1")
    df.head()
    df.rename(columns={"v1": "Label", "v2": "Text"}, inplace=True)

    with open(file_name, 'r') as f:
        column = (row[0] for row in csv.reader(f))
        commons = []
        x = 0
        commons.append(Counter(column).most_common(30))
    print(commons)
    comm=[]
    for a in range(30):
        comm.append(str(commons[0][a][0]))
        a=a+1

    print(comm)
    f.close()
    r = csv.reader(open('labeled_suc.csv'))  # Here your csv file
    lines = list(r)
    count_row = df.shape[0]
    a = 0
    for a in range(count_row):
        if lines[a][0] not in comm:
            lines[a][0] = 'other'
        a=a+1

    with open('modified.csv', 'w', newline='', errors='ignore') as file:
        writer = csv.writer(file)
        writer.writerows(lines)
    file.close()




if __name__ == '__main__':
    main()
