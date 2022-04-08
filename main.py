from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, QuadgramCollocationFinder
import string
import json
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
import os
from nltk.metrics.association import QuadgramAssocMeasures
import yake

## GROUP 11 ##

# defining the stopwords
stop_words = stopwords.words('turkish')
# path of the input json folder
path = "2021-01-20220322T055600Z-001\\2021-01\\"
# count the number of json file ( for different input folders)
dir_path = r'2021-01-20220322T055600Z-001\\2021-01\\'
num_of_file = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
print(num_of_file)
# bigram output
file1 = open("bigram_out.txt", "w", encoding="utf-8")

# trigram output
file2 = open('trigram_out.txt', 'w', encoding="utf-8")
file3 = open('quadgram_out.txt', 'w', encoding="utf-8")
file4 = open('keyword.txt', 'w', encoding="utf-8")

language = "tr"
max_ngram_size = 3
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 10

i = 1
while i < num_of_file + 1:
    fpath = path + str(i) + ".json"

    with open(fpath, encoding='utf-8') as fh:
        data = json.load(fh)
    single_text = ""

    single_text = single_text + str(data["Dairesi"] + " ")
    single_text = single_text + str(data["Mahkemesi"] + " ")
    single_text = single_text + str(data["Mahkeme Günü"] + " ")
    single_text = single_text + str(data["Mahkeme Ayı"] + " ")
    single_text = single_text + str(data["Mahkeme Yılı"] + " ")
    single_text = single_text + str(data["Suç"] + " ")
    single_text = single_text + str(data["Dosyanın Daireye Geliş Günü"] + " ")
    single_text = single_text + str(data["Dosyanın Daireye Geliş Ayı"] + " ")
    single_text = single_text + str(data["Dosyanın Daireye Geliş Yılı"] + " ")
    single_text = single_text + str(data["Kanun Yolu"] + " ")
    single_text = single_text + str(data["Temyiz Eden"] + " ")
    single_text = single_text + str(data["Dava Türü"] + " ")
    single_text = single_text + str(data["Birinci Mahkemesi"] + " ")
    single_text = single_text + str(data["ictihat"] + " ")

    # all lowercase for detecting correctly
    single_text = single_text.lower()

    # extracting keywords part
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                                features=None)

    line4 = "Keywords of json" + str(i) + " : ", custom_kw_extractor.extract_keywords(single_text)

    file4.write(str(line4) + "\n")

    # tokenize the text
    words = word_tokenize(single_text)
    # remove stopwords
    words = [word for word in words if word not in stop_words]
    # remove punctuations
    punctuations = list(string.punctuation)

    words = [word for word in words if word not in punctuations]

    # bigram method
    bigram_collocation = BigramCollocationFinder.from_words(words)
    # print("Bigrams:", bigram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 10))
    line1 = "Bigrams of json" + str(i) + " : ", bigram_collocation.nbest(BigramAssocMeasures.likelihood_ratio, 10)

    file1.write(str(line1) + "\n")
    # trigram method
    trigram_collocation = TrigramCollocationFinder.from_words(words)
    # print("Trigrams:", trigram_collocation.nbest(TrigramAssocMeasures.likelihood_ratio, 10))
    line2 = "Trigrams of json" + str(i) + " : ", trigram_collocation.nbest(TrigramAssocMeasures.likelihood_ratio, 10)
    file2.write(str(line2) + "\n")
    # quadgram method
    quadgram_collocation = QuadgramCollocationFinder.from_words(words)
    line3 = "Quadgrams of json" + str(i) + " : ", quadgram_collocation.nbest(QuadgramAssocMeasures.likelihood_ratio, 10)
    file3.write(str(line3) + "\n")

    i = i + 1
file1.close()
file2.close()
file3.close()
file4.close()
