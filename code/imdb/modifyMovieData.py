import os
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize

for movieDir in ('train/pos', 'train/neg', 'train/unsup', 'test/pos', 'test/neg'):
    with open(os.path.join(movieDir + '_mod', 'files'), 'w') as ff:
        for fn in os.listdir(movieDir):
            print(os.path.join(os.path.join(os.path.abspath('.'),
                                            movieDir + '_mod'),
                               fn),
                  file=ff)
            with open(os.path.join(movieDir,fn)) as f:
                lines = f.read().splitlines()
            for s in lines:
                s = s.replace('<br />', ' ')
                with open(os.path.join(movieDir + '_mod', fn), 'w') as f:
                    for sent in sent_tokenize(s):
                        print(sent, file=f)
