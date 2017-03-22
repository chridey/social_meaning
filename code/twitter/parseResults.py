import sys

from scipy.stats.distributions import binom

name = None
train = None
test = None
firstTestScore = None
secondTestScore = None
testScoreAll = None

arr = []
with open(sys.argv[1]) as f:
    for line in f:
        if line.startswith('Namespace'):
            #extract filename
            name = line.split('filename=')[1].split(', loadModel=')[0]
        elif line.startswith('('):
            #extract train or test depending
            if train is None:
                train = int(line.replace('(', '').replace(',', '').split()[0])
                continue

            test = int(line.replace('(', '').replace(',', '').split()[0])
            
        elif line.startswith('Test score:'):
            if firstTestScore is None:
                firstTestScore = float(line.replace('Test score: ', ''))
                continue

            secondTestScore = float(line.replace('Test score: ', ''))
            assert(train is not None)
            assert(test is not None)
            assert(testScoreAll is not None)

            a_cdf = 1-binom.cdf(secondTestScore*test, test, firstTestScore)
            if firstTestScore > secondTestScore:
                a_cdf = -a_cdf
            b_cdf = 1-binom.cdf(secondTestScore*test, test, testScoreAll)
            if testScoreAll > secondTestScore:
                b_cdf = -b_cdf
            #print(a_cdf, b_cdf)
            arr.append([name, train, test, firstTestScore, secondTestScore, testScoreAll, a_cdf, b_cdf])
            
            name = None
            train = None
            test = None
            firstTestScore = None
            secondTestScore = None
            testScoreAll = None
        elif line.startswith('Test score all_model'):
            testScoreAll = float(line.replace('Test score all_model: ', ''))

for s in sorted(arr, key=lambda x:x[6], reverse=True):
    print(s)
