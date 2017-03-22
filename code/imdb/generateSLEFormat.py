import os
import sys
import math
import argparse
import subprocess
import itertools
from collections import defaultdict

#import numpy
from nltk.tokenize import sent_tokenize,word_tokenize

sys.path.insert(0, '.')
from featureExtractor import FeatureExtractor
from dataPoint import DataPoint
sys.path.insert(0, '/local/nlp/chidey/social_meaning/')
from trainDiscourseSentiment import pair_tokenizer

def main(dataDir, fe, session, numtrain, hidden=None, printFeatures=True):
    movieDir = dataDir

    hiddenFileName = 'hidden_vars_' + session
    trainfile = open('trainfile_' + session, 'w')
    validatefile = open('validatefile_' + session, 'w')
    testfile = open('testfile_' + session, 'w')

    if not hidden:
        for partition in ('train', 'validate', 'test'):
            try:
                os.unlink('{}_{}file'.format(hiddenFileName, partition))
            except OSError:
                pass
    else:
        latentSubjectivity = defaultdict(list)
        for partition in ('train', 'validate', 'test'):
            with open('{}_{}file'.format(hidden, partition)) as f:
                for line in f:
                    line = line.strip()
                    latentSubjectivity[partition].append(set([int(i) for i in line.split()][1:]))

    for whichData in ('train', 'test'):

        baseDir = os.path.join(movieDir, whichData)

        totalFiles = 0
        for polarityDir,sentiment in (('neg_mod',-1), ('pos_mod', 1)):
            if whichData == 'test':
                outfile = testfile
            else:
                outfile = trainfile

            finalSubjectivity = []
            absMovieDir = os.path.join(baseDir,
                                       polarityDir)

            with open(os.path.join(absMovieDir, 'files')) as bf:
                fileIndex = 0
                for fn in bf:
                    fn = fn.strip()
                    print(fn, file=sys.stderr)
                    with open(fn) as f:
                        lines = f.read()
                        sentences = []
                        subjectivity = set()

                        with open(os.path.join(fn + '_auto_anns',
                                               'sent_subj.txt')) as sf:
                            for line in sf:
                                #lines look like pos_mod_6609_8.txt_0_99 obj
                                fileInfo, subj = line.split()
                                if subj == 'subj':
                                    subjectivity.add(len(sentences))

                                indices = fileInfo.split('_')
                                s,f = int(indices[-2]),int(indices[-1])
                                sentences.append(lines[s:f])
                                #print(subjectivity, sentences)
                                
                        if len(sentences) == 0:
                            continue
                        if hidden:
                            partition = whichData
                            offset = 0
                            if partition == 'train':
                                if len(finalSubjectivity) >= numtrain:
                                    partition = 'validate'
                                    if totalFiles > 0:
                                        offset = 2*numtrain
                                    else:
                                        offset = numtrain
                                else:
                                    offset = max(0,totalFiles-numtrain)
                            subjectivity = latentSubjectivity[partition][totalFiles+fileIndex-offset]
                            #print(partition,totalFiles,fileIndex,offset,subjectivity,len(sentences))
                            assert(len(sentences)>max(subjectivity))

                        fileIndex+=1
                        if len(subjectivity) == 0 or (-1 in subjectivity and len(subjectivity-{-1}) == 0):
                            #just use the first sentence
                            #continue
                            #subjectivity.add(0)
                            #alternatively, use all sentences
                            subjectivity |= {i for i in range(len(sentences))}

                        tokenizedSentences = [[word.lower() for word in word_tokenize(sentence)] for sentence in sentences]
                        globalInfo = fe.addGlobalInfo(tokenizedSentences)

                        print('{} {}'.format(sentiment, len(sentences)), file=outfile)
                        docIndices = set()
                        prevTokens = None
                        for sentenceIndex,tokens in enumerate(tokenizedSentences):
                        #for sentenceIndex,sentence in enumerate(sentences):
                            #tokens = [word.lower() for word in word_tokenize(sentence)]
                            curr = (sentenceIndex in subjectivity)
                            prev = (sentenceIndex-1 in subjectivity)
                            dp = DataPoint({'curr_words': tokens,
                                            'prev_words': prevTokens,
                                            'curr_index': sentenceIndex,
                                            'filename': fn,
                                            'curr_subj': curr,
                                            'prev_subj': prev,
                                            'global_info': globalInfo})

                            wordFeatures = fe.addFeatures(dp, fe.lexicalFeatures)
                            otherFeatures = fe.addFeatures(dp, fe.polarityFeatures)

                            #magnitude = (math.sqrt(sum(i**2 for i in itertools.chain(
                            #    wordFeatures.values(),
                            #    otherFeatures.values()))))
                            magnitude = math.sqrt(len(wordFeatures.keys()))# + len(otherFeatures.keys()))
                            print(sentenceIndex, end=' ', file=outfile)
                            for index in sorted(set(wordFeatures) | set(otherFeatures)):
                                if index in wordFeatures:
                                    print('{}:{}'.format(index, 1.0/magnitude), end=' ', file=outfile)
                                else:
                                    print('{}:{}'.format(index, otherFeatures[index]), end=' ', file=outfile)                              
                            subjectivityFeatures = fe.addFeatures(dp,
                                                                  fe.subjectivityFeatures)
                            if 'markers' in fe.subjectivityFeatures:
                                markerFeatures = fe.addFeatures(dp,
                                                                {'markers'})
                            else:
                                markerFeatures = {}
                            subjectivityFeatures = fe.addFeatures(dp,
                                                                  fe.subjectivityFeatures-{'markers'})

                            #magnitude = (math.sqrt(sum(i**2 for i in itertools.chain(
                            #    wordFeatures.values(),
                            #    subjectivityFeatures.values()))))
                            magnitude = math.sqrt(len(wordFeatures.keys()) + len(markerFeatures.keys()))
                            for index in sorted(set(wordFeatures) | set(markerFeatures) | set(subjectivityFeatures)):
                                if index in subjectivityFeatures:
                                    print('S{}:{}'.format(index, subjectivityFeatures[index]), end=' ', file=outfile)                       
                                else:
                                    print('S{}:{}'.format(index, 1.0/magnitude), end=' ', file=outfile)                       
                            print(file=outfile)
                            docIndices.update(set(wordFeatures))
                            prevTokens = tokens

                        #now print document level features
                        magnitude = math.sqrt(len(docIndices))
                        print(len(sentences), end=' ', file=outfile)
                        docFeatures = fe.addDocumentFeatures(dp)
                        for index in sorted(docIndices | set(docFeatures)):
                            if index in docIndices:
                                print('{}:{}'.format(index, 1.0/magnitude), end=' ', file=outfile)
                            else:
                                print('{}:{}'.format(index, docFeatures[index]), end=' ', file=outfile)
                        print("\n", file=outfile)

                        subjectiveString = '{} '.format(len(subjectivity))
                        finalSubjectivity.append('{} '.format(len(subjectivity)) + \
                                                     ' '.join(str(i) for i in sorted(subjectivity)))

                        if whichData == 'train' and len(finalSubjectivity) == numtrain:
                            outfile = validatefile
                            if not hidden:
                                with open('{}_trainfile'.format(hiddenFileName), 'a') as hf:
                                    for line in finalSubjectivity:
                                        print(line, file=hf)

                        if len(finalSubjectivity) >= float('inf'):
                            break

            totalFiles += fileIndex

            if not hidden:
                if whichData == 'test':
                    with open('{}_testfile'.format(hiddenFileName), 'a') as hf:
                        for line in finalSubjectivity:
                            print(line, file=hf)
                else:
                    with open('{}_validatefile'.format(hiddenFileName), 'a') as hf:                  
                        for line in finalSubjectivity[numtrain:]:
                            print(line, file=hf)

    if printFeatures:
        fe.printFeatures(args.session)

    trainfile.close()
    validatefile.close()
    testfile.close()

parser = argparse.ArgumentParser(description='train and/or evaluate a classifier on a dataset with altlexes')

parser.add_argument('session',
                    help='required session name for creating output files')
                    
parser.add_argument('numtrain', type=int,
                    help='the number of examples to put in training (the rest in validation)')

parser.add_argument('--markersFile',
                    help='file of explicit discourse connectives to add features for')

parser.add_argument('--hidden', 
                    help='use SLE formatted file for hidden variables instead of opinion finder')

#parser.add_argument('--featuresFile', type=open,
#                    help='file with list of named features')

parser.add_argument('--iterations', type=int, default=0,
                    help='number of iterations to run latent SVM')

parser.add_argument('--subjectivityFeatures',
                    help='comma-separated list of named features')

parser.add_argument('--polarityFeatures',
                    help='comma-separated list of named features')

parser.add_argument('--dataDir',
                    help='data directory')

parser.add_argument('-c', type=float, default=1000)
parser.add_argument('-l', type=int, default=50)
parser.add_argument('--saveCache', action='store_true')

args = parser.parse_args()
print(args)

featureSettings = {}
if args.subjectivityFeatures is not None:
    print(args.subjectivityFeatures, len(args.subjectivityFeatures))
    if not args.subjectivityFeatures or args.subjectivityFeatures == '0':
        featureSettings['subjectivity'] = set()
    else:
        featureSettings['subjectivity'] = set(i for i in args.subjectivityFeatures.split(','))

if args.polarityFeatures is not None:
    if not args.polarityFeatures or args.polarityFeatures == '0':
        featureSettings['polarity'] = set()
    else:
        featureSettings['polarity'] = set(i for i in args.polarityFeatures.split(','))
        
#elif args.featuresFile:
#    features = set(args.featuresFile.read().splitlines())
#else:
#    features = {}
#print(features)

if args.markersFile:
    featureSettings['markersFile'] = args.markersFile
    #hiddenFileName += '_markers'

print(featureSettings)
fe = FeatureExtractor(**featureSettings)

os.mkdir(args.session)
os.chdir(args.session)

if not args.dataDir:
    dataDir = '/local/nlp/chidey/social_meaning/aclImdb/'
else:
    dataDir = args.dataDir
main(dataDir, fe, '{}_{}'.format(args.session,0), args.numtrain)

learnCommandTemplate='/local/nlp/chidey/social_meaning/yessenalina/sle_movieReviews/bin/svm_sle_learn -v 3 -c {0} -l {1} {2}_{3}_{4} hidden_vars_{3}_{4}_{2} model_{3}_{4}'.format(args.c, args.l, '{0}', args.session, '{1}')
#2=train/validate/testfile
#3=session
#4=iteration
classifyCommandTemplate = '/local/nlp/chidey/social_meaning/yessenalina/sle_movieReviews/bin/svm_sle_classify {0} {1} model_{2}_{3} {4}'.format('{0}', '{1}', args.session, '{2}', '{3}')
#0=latent flag
#1=train/validate/testfile
#2=session
#3=iteration
#4=output file name

#train the model on the current training file
learnCommand = learnCommandTemplate.format('trainfile', 0)
print(learnCommand)
p = subprocess.Popen(learnCommand.split()) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.communicate()

#classify the validation set
#classify the testing set
for partition in 'validatefile', 'testfile':
    classifyCommand = classifyCommandTemplate.format('', '{}_{}_{}'.format(partition, args.session,0), 0, 'junk')
    print(classifyCommand)
    p = subprocess.Popen(classifyCommand.split()) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    
for i in range(1,args.iterations+1):
    #use the current model to create hidden variables for train, validate, and test
    hiddenFileName = 'hidden_vars_{}_{}'.format(args.session, i)
    for partition in 'trainfile', 'validatefile', 'testfile':
        #classifyCommand = classifyCommandTemplate.format('-l', partition, i-1, '{}_{}'.format(hiddenFileName, partition))
        classifyCommand = classifyCommandTemplate.format('-l', '{}_{}_{}'.format(partition, args.session,i-1), i-1, '{}_{}'.format(hiddenFileName, partition))
        print(classifyCommand)
        p = subprocess.Popen(classifyCommand.split()) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()

    #generate the new files with the new hidden variables file
    main(dataDir, fe, '{}_{}'.format(args.session,i), args.numtrain, hidden=hiddenFileName, printFeatures=False)
    
    #re-train the model using the new training file
    learnCommand = learnCommandTemplate.format('trainfile', i)
    print(learnCommand)
    p = subprocess.Popen(learnCommand.split()) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()

    #classify the validation set using the previous model
    #classify the testing set using the previous model
    for partition in 'validatefile', 'testfile':
        #classifyCommand = classifyCommandTemplate.format('', partition, i, 'junk')
        classifyCommand = classifyCommandTemplate.format('', '{}_{}_{}'.format(partition, args.session,i), i-1, 'junk')
        print(classifyCommand)
        p = subprocess.Popen(classifyCommand.split()) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()

if args.saveCache:
    fe.saveCache()

#initialize index lookup table with discourse features
#read in list of files
#for each file
# read in file as single string
# read in opinion finder sentences (sent_subj.txt)
# append subjectivity indices to file
# extract sentences according to opinion finder
# output overall score
# for each sentence
#  for each word
#     look up the index or add it to the lookup table
#  normalize vectors to be unit length
#  output sentiment features as index:weight and subjective features as Sindex:weight
#  subjective features will also include discourse features
#
#output overall document features

#iteratively train SVM?
