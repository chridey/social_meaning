from __future__ import print_function

import sys
import os

from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize,word_tokenize

connectivesFile = sys.argv[1]
reviewDir = sys.argv[2]

with open(connectivesFile,encoding='latin_1') as f:
    connectives = f.read().splitlines()

output = defaultdict(list)
for fn in os.listdir(reviewDir):
    print(fn)
    with open(os.path.join(reviewDir,fn),encoding='latin_1') as f:
        for line in f:
            line = line.replace('<br />',' ')
            for sentence in sent_tokenize(line):
                words = word_tokenize(sentence)
                for connective in connectives:
                    if connective.count(' ') == 0:
                        if connective in words:
                            output[connective].append(sentence)
                    else:
                        if connective in sentence:
                            output[connective].append(sentence)

for connective in output:
    with open(connective + '.sentences', 'w') as f:
        for sentence in output[connective]:
            print(sentence,file=f)
