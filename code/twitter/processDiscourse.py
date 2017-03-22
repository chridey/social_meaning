import csv

from collections import defaultdict

from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

markersFile = '/local/nlp/chidey/social_meaning/aclImdb/markers'
sn = SnowballStemmer("english")

with open(markersFile) as mf:
    markers = set(mf.read().splitlines())
    maxMarkerLength = max(len(i.split()) for i in markers)
    
counts = [defaultdict(int) for i in range(len(markers)*2)]
indexLookup = {}
fileList = set()

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
                with open('new_tokens_all_{}'.format(sentiment), 'a') as f:
                    print(' '.join(tokens), file=f)
                continue

            tokens1 = tokens[:bestMarkerIndex]
            tokens2 = tokens[bestMarkerIndex+bestMarkerLength:]
            #print(sentiment, text, tokens, tokens1, tokens2)            

            if not len(tokens1) or not len(tokens2):
                with open('new_tokens_all_{}'.format(sentiment), 'a') as f:
                    print(' '.join(tokens), file=f)
                continue
            marker = '_'.join(tokens[bestMarkerIndex:bestMarkerIndex+bestMarkerLength])

            '''
            output1 = ' '.join(tokens1).replace('!', ' , ').replace('.', ' , ').replace('?', ' , ')
            output2 = ' '.join(tokens2).replace('!', ' , ').replace('.', ' , ').replace('?', ' , ')
            with open('split_corenlp_{}_{}'.format(marker, sentiment), 'a') as f:
                print(output1 + ' .', file=f)
                print(output2 + ' .', file=f)
            output = ' '.join(tokens).replace('!', ' , ').replace('.', ' , ').replace('?', ' , ')
            with open('corenlp_{}_{}'.format(marker, sentiment), 'a') as f:
                print(output + '.', file=f)

            '''
            with open('new_tokens_{}_{}'.format(marker, sentiment), 'a') as f:
                #print(' '.join(tokens1), file=f, end='\t')
                #print(' '.join(tokens2), file=f)
                print('\t'.join([' '.join(tokens1),
                                 marker,
                                 ' '.join(tokens2)]),
                      file=f)
            continue            
                
            sentiment = 'all'
            continue
                
            for token1 in tokens1:
                for token2 in tokens2:
                    a = '{}\t{}'.format(' '.join(tokens[bestMarkerIndex:bestMarkerIndex+bestMarkerLength]),
                                               sentiment)
                    b = '{}\t{}'.format(token1, token2)
                    #print(a,b)
                    if a not in indexLookup:
                        indexLookup[a] = len(indexLookup)

                    try:
                        counts[indexLookup[a]][b]+=1
                    except IndexError as e:
                        print(a,b, indexLookup[a], sentence)
                        print(e)
                        raise Exception

        print(irow)
        #if irow>100:
        #    break

exit()
vec = DictVectorizer()
transformer = TfidfTransformer()

countsArray = vec.fit_transform(counts)
countsTransformed = transformer.fit_transform(countsArray)

reverseLookup = {j:i for i,j in indexLookup.items()}
wordPairs = vec.get_feature_names()

print (type(countsTransformed), countsTransformed.shape)
#now write out counts to separate files
for i,a in enumerate(countsTransformed):
    try:
        markerSentiment = reverseLookup[i]
    except KeyError:
        break
    print(i, markerSentiment, type(a), a.shape)

    with open(markerSentiment, 'w') as f:
        for j,c in enumerate(a.toarray()[0]):
            #print(j, wordPairs[j])
            if c > 0:
                print('{}\t{}'.format(wordPairs[j], c), file=f)

                            
