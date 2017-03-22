import csv

from collections import defaultdict

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from scipy.stats import chi2_contingency, chisquare

markersFile = '/local/nlp/chidey/social_meaning/aclImdb/markers'
sn = SnowballStemmer("english")

with open(markersFile) as mf:
    markers = set(mf.read().splitlines())
    maxMarkerLength = max(len(i.split()) for i in markers)
    
counts = {}

#run two chi squared tests
#1) for each discourse marker, compare positive and negative polarity to each other
#2) for each discourse marker, compare positive/negative to expected positie/negative

with open("Sentiment Analysis Dataset.csv", newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for irow,row in enumerate(csvreader):
        sentiment, text = row[1],row[3]
        for sentence in sent_tokenize(text):
            tokens = [sn.stem(word.lower()) for word in word_tokenize(sentence)]
            minDiff = float('inf')
            bestMarkerIndex = None
            bestMarkerLength = 0

            for index,token in enumerate(tokens):
                for i in range(maxMarkerLength):
                    marker = ' '.join(tokens[index:index+i+1])
                    if marker in markers:
                        diff = abs(len(tokens)-2*index)
                        if diff < minDiff:
                            minDiff = diff
                            bestMarkerIndex = index
                            bestMarkerLength = i+1

            if bestMarkerIndex is None:
                continue
            
            tokens1 = tokens[:bestMarkerIndex]
            tokens2 = tokens[bestMarkerIndex+bestMarkerLength:]
            #print(sentiment, text, tokens, tokens1, tokens2)

            if not len(tokens1) or not len(tokens2):
                continue

            marker = '_'.join(tokens[bestMarkerIndex:bestMarkerIndex+bestMarkerLength])
            if sentiment not in counts:
                counts[sentiment] = {}
            if marker not in counts[sentiment]:
                counts[sentiment][marker] = {}
            for token1 in tokens1:
                if "1_".format(token1) not in counts[sentiment][marker]:
                    counts[sentiment][marker]["1_"+token1] = 0
                counts[sentiment][marker]["1_"+token1]+=1
            for token2 in tokens2:
                if "2_{}".format(token2) not in counts[sentiment][marker]:
                    counts[sentiment][marker]["2_"+token2] = 0
                counts[sentiment][marker]["2_"+token2]+=1

        print(irow)
        #if irow>100000:
        #    break

abs_counts = {}
totalMarker = {}
for sentiment in counts:
    abs_counts[sentiment] = {}
    totalMarker[sentiment] = {}
    for marker in counts[sentiment]:
        totalMarker[sentiment][marker] = 0
        for token in counts[sentiment][marker]:
            if token not in abs_counts[sentiment]:
                abs_counts[sentiment][token]=0
            abs_counts[sentiment][token] += counts[sentiment][marker][token]
            totalMarker[sentiment][marker] += counts[sentiment][marker][token]
            
total = {}
for sentiment in abs_counts:
    total[sentiment] = sum(abs_counts[sentiment].values())

for sentiment in counts:
    for marker in counts[sentiment]:
        #print(sentiment, marker, totalMarker[sentiment][marker],total[sentiment])
        observedRows = []
        expectedRows = []
        for token in counts[sentiment][marker]:
            #print("\t",token, counts[sentiment][marker][token], abs_counts[sentiment][token])
            observed = counts[sentiment][marker][token]
            expected = totalMarker[sentiment][marker]*abs_counts[sentiment][token]/total[sentiment]

            if observed < 5 or expected < 5:
                continue
            #print(token, observed, expected)
            observedRows.append(observed)
            expectedRows.append(expected)
            
        chisq, p = chisquare(observedRows, expectedRows)
        print(sentiment, marker, len(observedRows), chisq, p)
