from __future__ import print_function

import sys
import argparse
import itertools

from pprint import pprint
from time import time
import logging

import numpy as np

from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Binarizer
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn import cross_validation
from sklearn.externals import joblib

from scipy.sparse import vstack,hstack

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
#        ('vect', CountVectorizer()),
#        ('tfidf', TfidfTransformer()),
    ('binary', Binarizer()),
#    ('chi', SelectKBest(chi2)),
    ('clf', SGDClassifier()),
    ])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
#    'chi__k': (10000, 100000, 'all'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }

'''
Parameters of the estimators in the pipeline can be accessed using the <estimator>__<parameter> syntax:
    >>>
    >>> clf.set_params(svm__C=10)
    Pipeline(steps=[('reduce_dim', PCA(copy=True, n_components=None,
                                           whiten=False)), ('svm', SVC(C=10, cache_size=200, class_weight=None,
                                                                           coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=-1,
                                                                           probability=False, random_state=None, shrinking=True, tol=0.001,
                                                                           verbose=False))])
    This is particularly important for doing grid searches:
        >>>
        >>> from sklearn.grid_search import GridSearchCV
        >>> params = dict(reduce_dim__n_components=[2, 5, 10],
                          ...               svm__C=[0.1, 10, 100])
        >>> grid_search = GridSearchCV(clf, param_grid=params)

vstack([x, x]).toarray()


binarizer = preprocessing.Binarizer(threshold=1)
binarizer.transform(X)

pipeline = []
parameters = {}
'''

class FileIterator:
    def __init__(self, fn1, fn2, maxCount=float('inf'), numTest=0):
        self.fn1 = fn1
        self.fn2 = fn2
        self.fn1Lines = 0
        self.fn2Lines = 0
        self.maxCount = maxCount
        self.numTest = numTest
        
    def iterFiles(self, start=0, maxCount=None):
        self.fn1Lines = 0
        self.fn2Lines = 0

        if maxCount is None:
            maxCount = self.maxCount

        with open(self.fn1) as f:
            for line in f:
                if self.fn1Lines < start:
                    continue
                if self.fn1Lines > maxCount:
                    break
                self.fn1Lines+=1
                yield line
        with open(self.fn2) as f:
            for line in f:
                if self.fn2Lines < start:
                    continue
                if self.fn2Lines > maxCount:
                    break
                self.fn2Lines+=1
                yield line

def split_tokenizer(s):
    part1, part2 = s.split('\t')
    return s.split() + ['1-'+i for i in part1.split()] + ['2-'+i for i in part2.split()]

def pair_tokenizer(s):
    part1, part2 = s.split('\t')
    return itertools.chain(s.split(), iter(' '.join(i) for i in itertools.product(part1.split(), part2.split())))

def marker_aware_tokenizer(s):
    if '\t' in s:
        tokens1, marker, tokens2 = s.split('\t')
        return itertools.chain(tokens1.split(), tokens2.split(), iter(marker + ' '.join(i) for i in itertools.product(tokens1.split(), tokens2.split())))
    else:
        return s.split()

topK = {'next', 'now', 'yet', 'still', 'as', 'but', 'or', 'when', 'then'}
def topk_marker_aware_tokenizer(s):
    if '\t' in s:
        tokens1, marker, tokens2 = s.split('\t')
        if marker in topK:
            return itertools.chain(tokens1.split(), tokens2.split(), iter(marker + ' '.join(i) for i in itertools.product(tokens1.split(), tokens2.split())))
        else:
            return s.split()
    else:
        return s.split()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train and/or evaluate a classifier on a dataset')

    parser.add_argument('filename',
                        help='required first part of filename')
    parser.add_argument('--test')
    parser.add_argument('--numBalance', type=int,
                        help='the number of examples to put in training for each class')

    parser.add_argument('--numTest', type=int, default=0,
                        help='the number of examples to set aside for testing')

    parser.add_argument('--saveModel')
    parser.add_argument('--loadModel')

    parser.add_argument('--tokenizer', choices=('default', 'split', 'pair', 'marker', 'topk'),
                        default='default')

    parser.add_argument('--corenlp', action='store_true')
                        
    args = parser.parse_args()
    print(args)
    
    filename = args.filename
    maxCount = args.numBalance
    
    #pipeline should not include the count vectorizer so we can do separate feature extraction
    #first call countvectorizer on negative sentiment, then positive sentiment
    if args.tokenizer == 'split':
        tokenizer=split_tokenizer
    elif args.tokenizer == 'pair':
        tokenizer=pair_tokenizer
    elif args.tokenizer == 'marker':
        tokenizer=marker_aware_tokenizer
    elif args.tokenizer == 'topk':
        tokenizer=topk_marker_aware_tokenizer
    else:
        tokenizer=str.split

    vectorizer = CountVectorizer(min_df=1, tokenizer=tokenizer)
    if not args.test:
        fi = FileIterator(filename + "_0", filename + "_1", maxCount, args.numTest)
        X = vectorizer.fit_transform(fi.iterFiles())
        y = np.array([0] * fi.fn1Lines + [1] * fi.fn2Lines)

        #vstack([x0, x1])
        #hstack for any additional features

        if args.numTest:
            percentage = 1.0*args.numTest/len(y)
        else:
            percentage = 0

        if percentage > .5:
            percentage = .5

        print(percentage)

        #X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        #    X, y, test_size=percentage, random_state=0)

        #need to have balanced training and testing
        #this shuffles the data
        X, _, y, _ = cross_validation.train_test_split(
            X, y, test_size=percentage, random_state=0)
        train_indices, test_indices = list(cross_validation.StratifiedKFold(y, n_folds=int(1/percentage), random_state=0, shuffle=False))[0]
        X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    else:
        fi = FileIterator(filename + "_0", filename + "_1")
        X_train = vectorizer.fit_transform(fi.iterFiles())
        y_train = np.array([0] * fi.fn1Lines + [1] * fi.fn2Lines)
        #this shuffles the data
        X_train, _, y_train, _ = cross_validation.train_test_split(
            X_train, y_train, test_size=0, random_state=0)

        fi2 = FileIterator(args.test + "_0", args.test + "_1")
        X_test = vectorizer.transform(fi2.iterFiles())
        y_test = np.array([0] * fi2.fn1Lines + [1] * fi2.fn2Lines)

    if args.corenlp:
        train_set = set(train_indices)
        test_set = set(test_indices)
        with open('corenlp_train_{}'.format(filename), 'w') as f:
            for index,line in enumerate(fi.iterFiles()):
                if index in train_set:
                    #print(index)
                    output1,output2 = line.split('\t')
                    for char in ("~!@#$%^&*()_-+={}|[];:,./<>?"):
                        output1 = output1.replace(char, ' , ')
                        output2 = output2.replace(char, ' , ')

                    output1 = output1[:-1]+'.'
                    print(output1, file=f)
                    output2 = output2[:-1]+'.'
                    print(output2, file=f)
                    
        with open('corenlp_test_{}'.format(filename), 'w') as f:
            for index,line in enumerate(fi.iterFiles()):
                #assume filename starts with tokens_{marker}
                if index in test_set:
                    #print(index)
                    _,marker = filename.split('_')
                    marker = marker.replace('_', ' ')
                    output1,output2 = line.split('\t')
                    output = ' '.join([output1, marker, output2])
                    #output = output.replace('!', ' , ').replace('.', ' , ').replace('?', ' , ')
                    for char in ("~!@#$%^&*()_-+={}|[];:,./<>?"):
                        output = output.replace(char, ' , ')

                    output = output[:-1]+'.'
                    print(output, file=f)
        exit()

    #TODO: select best with chi squared as part of pipeline

    print(X_train.shape)
    print(X_test.shape)
    #print(y_train)
    #print(y_test)
    #pipeline should consist of binarizer (or maybe do tf-idf ngram selection) and classifier
    #pipeline = make_pipeline(Binarizer(), SGDClassifier())
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    #test model if args.numTest
    if args.numTest or args.test:
        print("Test score: %.03f" % grid_search.score(X_test, y_test))
        if args.loadModel:
            _vectorizer = joblib.load(args.loadModel + '_vectorizer')
            X = _vectorizer.transform(fi.iterFiles())
            y = [0] * fi.fn1Lines + [1] * fi.fn2Lines
            X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                X, y, test_size=percentage, random_state=0)
            model = joblib.load(args.loadModel)
            print("Test score %s: %.03f" % (args.loadModel,
                                            model.score(X_test, y_test)))


    #save if args.output
    if args.saveModel:
        joblib.dump(grid_search, args.saveModel)
        joblib.dump(vectorizer, args.saveModel + '_vectorizer')
