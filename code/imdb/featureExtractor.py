import sys
import os
import itertools

from collections import defaultdict

from sklearn.externals import joblib

from chnlp.altlex.extractSentences import MarkerScorer

sys.path.insert(0, '/local/nlp/chidey/social_meaning/')
from trainDiscourseSentiment import pair_tokenizer 

class MarkerPolarityScorer(MarkerScorer):
    filePath = '/local/nlp/chidey/social_meaning/'
    
    def __init__(self):
        super().__init__()
        self.markerPairs = {}

    def iterSubset(self):
        for polarity in (0,1):
            yield polarity
        
    def scoreWords(self, words1, words2, marker):
        dotProducts = {}

        #for polarity in (0,1):
        for polarity in self.iterSubset():
            modMarker = "\t".join((marker, str(polarity))) #'{}\t{}'.format(marker, polarity)

            if modMarker not in self.markerPairs:
                try:
                    print("opening {}".format(modMarker))
                    self.markerPairs[modMarker] = self.loadPairs(self.filePath + modMarker)
                except IOError: #FileNotFoundError:
                    print("cant open {}".format(modMarker))
                    continue
                except Exception:
                    print('unknown exception in MarkerPolarityScorer')
                    continue
                if len(self.markerPairs[modMarker]):
                    self.markerLengths[modMarker] = self.magnitude(self.markerPairs[modMarker])
                else:
                    del self.markerPairs[modMarker]

            instance = defaultdict(dict)
            for word1 in words1:
                for word2 in words2:
                    instance[word1][word2] = 1

            instanceLength = self.magnitude(instance)

            dotProducts[polarity] = self.dotProduct(instance,
                                                    self.markerPairs[modMarker]) / \
                                                    (self.markerLengths[modMarker] * \
                                                     instanceLength)

        return dotProducts

class MarkerSubjectivityScorer(MarkerPolarityScorer):
    def iterSubset(self):
        yield 'all'

class FeatureExtractor:
    scoresFile = '/local/nlp/chidey/cosine_scores'
    pscoresFile = '/local/nlp/chidey/polarity_cosine_scores'
    markersFile = '/local/nlp/chidey/social_meaning/aclImdb/markers'
    modelDir = '/local/nlp/chidey/social_meaning/'
    
    def __init__(self, subjectivity=None, polarity=None, markersFile=None):
        self._lexicalFeatures = {'bag_of_words'}
        if polarity is not None:
            self._polarityFeatures = polarity
        else:
            self._polarityFeatures = {}
        if subjectivity is not None:
            self._subjectivityFeatures = subjectivity
        else:
            self._subjectivityFeatures = {'marker_cosine',
                                          'markers'}
        
        self.ms = None
        self.scoreLookup = {}
        self.readScores = False

        self.mps = None
        self.pscoreLookup = {}
        self.pmarkerLookup = {}
        self.readPscores = False

        self.mss = None

        self.discourseModels = {}
        self.model = None
        self.dms = {}

        self.allDiscourseModel = {}
        self.adms = {}
        
        if not markersFile:
            markersFile = self.markersFile

        with open(markersFile) as mf:
            self.markers = set(mf.read().splitlines())
            self.maxMarkerLength = max(len(i.split()) for i in self.markers)

        self.validFeatures = {'marker_cosine' : self.getMarkerCosineSim,
                              'bag_of_words': self.getBagOfWords,
                              'markers': self.getMarkers,
                              'marker_polarity_cosine': self.getMarkerPolarityCosineSim,
                              'marker_subjective_cosine': self.getSubjectiveObjectiveCosineSim,
                              'rel_sentence': self.getRelSentence,
                              'rel_sentence_length': self.getRelSentenceLength,
                              'discourse_model': self.getDiscourseModelScore,
                              'all_discourse_model': self.getAllDiscourseModelScore}

        self.functionFeatures = dict((v,k) for k,v in self.validFeatures.items())

        self.featureLookup = {}

    @property
    def lexicalFeatures(self):
        return self._lexicalFeatures

    @property
    def polarityFeatures(self):
        return self._polarityFeatures

    @property
    def subjectivityFeatures(self):
        return self._subjectivityFeatures

    def addGlobalInfo(self, sentences):
        return {'num_sentences': len(sentences),
                'max_sentence_length': max(len(s) for s in sentences)}

    def _getFeatureIndex(self, feature):
        if feature not in self.featureLookup:
            self.featureLookup[feature] = len(self.featureLookup)
        return self.featureLookup[feature]

    def getRelSentence(self, dp):
        return {self._getFeatureIndex(self.functionFeatures[self.getRelSentence]) : dp.relIndex}
    def getRelSentenceLength(self, dp):
        return {self._getFeatureIndex(self.functionFeatures[self.getRelSentenceLength]) : dp.relLength}
        
    def getBagOfWords(self, dp):
        ret = {}

        if dp.currWords is None:
            return ret

        for word in dp.currWords:
            ret[self._getFeatureIndex(word)] = 1

        return ret

    def getMarkers(self, dp):
        ret = {}

        same = (dp.currSubj==dp.prevSubj)
        for i in range(self.maxMarkerLength):
            marker = ' '.join(dp.currWords[:i+1])
            if marker in self.markers:
                ret[self._getFeatureIndex((marker,same))] = 1
                break

        return ret
        #magnitude = math.sqrt(len(wordIndices) + len(otherIndices))
                
    def getMarkerCosineSim(self, dp):
        ret = {}
        if dp.prevStems is None:
            return ret

        if not self.readScores and os.path.exists(self.scoresFile):
            print('reading ' + self.scoresFile)
            with open(self.scoresFile) as sf:
                for line in sf:
                    line = line.strip()
                    index, marker, score = line.split("\t")
                    if index not in self.scoreLookup:
                        self.scoreLookup[index] = {}               
                    self.scoreLookup[index][marker] = score
            self.readScores = True
            
        try:
            scores = self.scoreLookup[dp.index]
        except Exception:
            if self.ms is None:
                self.ms = MarkerScorer()
            scores = self.ms.scoreWords(dp.prevStems, dp.currStems)
            self.scoreLookup[dp.index] = scores
            
        same = (dp.currSubj==dp.prevSubj)
        
        for marker in scores:
            ret[self._getFeatureIndex((self.functionFeatures[self.getMarkerCosineSim] + ' ' + marker,
                 same))] = scores[marker]

        return ret

    def _findMarker(self, tokens):
        minDiff = float('inf')
        bestMarkerIndex = None
        bestMarkerLength = 0
        for index in range(len(tokens)):
            for i in range(self.maxMarkerLength):
                marker = ' '.join(tokens[index:index+i+1])
                if marker in self.markers:
                    diff = abs(len(tokens)-2*index)
                    if diff < minDiff:
                        minDiff = diff
                        bestMarkerIndex = index
                        bestMarkerLength = i+1

        if bestMarkerIndex is None:
            return None
        
        tokens1 = tokens[:bestMarkerIndex]
        tokens2 = tokens[bestMarkerIndex+bestMarkerLength:]
        if not len(tokens1) or not len(tokens2):
            return None
        marker = ' '.join(tokens[bestMarkerIndex:bestMarkerIndex+bestMarkerLength])

        return marker,tokens1,tokens2
    
    def getMarkerPolarityCosineSim(self, dp):
        ret = {}

        if not self.readPscores and os.path.exists(self.pscoresFile):
            print('reading ' + self.pscoresFile)
            with open(self.pscoresFile) as sf:
                for line in sf:
                    line = line.strip()
                    index, marker, polarity, score = line.split("\t")
                    if index not in self.pscoreLookup:
                        self.pscoreLookup[index] = {}               
                    self.pscoreLookup[index][polarity] = score
                    self.pmarkerLookup[index] = marker
            self.readPscores = True
            
        try:
            scores = self.pscoreLookup[dp.index]
            marker = self.pmarkerLookup[dp.index]
        except Exception:
            if self.mps is None:
                self.mps = MarkerPolarityScorer()

            #try to find a marker
            result = self._findMarker(dp.currStems)

            if result is None:
                return ret

            marker, tokens1, tokens2 = result
            scores = self.mps.scoreWords(tokens1, tokens2, marker)
            self.pscoreLookup[dp.index] = scores

        if len(scores)==2:
            score = scores[0]-scores[1]
            return {self._getFeatureIndex((self.functionFeatures[self.getMarkerPolarityCosineSim] + ' ' + marker)) :score}
            
        for polarity in scores:
            ret[self._getFeatureIndex((self.functionFeatures[self.getMarkerPolarityCosineSim] + ' ' + marker,
                 polarity))] = scores[polarity]

        return ret

    def getSubjectiveObjectiveCosineSim(self, dp):
        ret = {}

        if self.mss is None:
            self.mss = MarkerSubjectivityScorer() #need to modify to do all MarkerPolarityScorer()
        if self.ms is None:
            self.ms = MarkerScorer()
            
        #try to find a marker
        result = self._findMarker(dp.currStems)

        if result is None:
            return ret

        marker, tokens1, tokens2 = result
        scores = self.mss.scoreWords(tokens1, tokens2, marker)
        if 'all' not in scores:
            return {}
        scores1 = {marker: scores['all']}
        #scores1 = {marker: self.mss.scoreWords(tokens1, tokens2, marker)['all']}
        scores2 = self.ms.scoreWords(tokens1, tokens2, {marker}) #modify scorer to take in marker
        #print(scores1, scores2)
        if marker not in scores1 or marker not in scores2:
            return {}

        if scores1[marker] > scores2[marker]:
            identifier = 'subj'
        else:
            identifier = 'obj'
            
        return {self._getFeatureIndex((self.functionFeatures[self.getSubjectiveObjectiveCosineSim] + ' ' + marker, identifier)): abs(scores1[marker]-scores2[marker])}

    def getDiscourseModelScore(self, dp):
        #extract marker
        #try to find a marker
        result = self._findMarker(dp.currStems)

        if result is None:
            return {}

        marker, tokens1, tokens2 = result

        topK = {'next', 'now', 'yet', 'still', 'as', 'but', 'or', 'when', 'then'}
        if marker not in topK:
            return {}

        if marker not in self.discourseModels:
            #try load model and vectorizer
            self.discourseModels[marker] = {}
            print('loading model ', marker)
            try:
                self.discourseModels[marker]['model'] = joblib.load(os.path.join(self.modelDir,
                                                                                 'model_tokens_' + marker))
            except Exception as e:
                print('could not open model ' + marker, e)
                self.discourseModels[marker] = None
            print('loading model vectorizer ', marker)
            try:
                self.discourseModels[marker]['vectorizer'] = joblib.load(os.path.join(self.modelDir,
                                                                                 'model_tokens_' + marker + '_vectorizer'))
            except Exception as e:
                print('could not open model vectorizer ' + marker, e)
                self.discourseModels[marker] = None

        if self.discourseModels[marker] is None:
            return {}

        if dp.index in self.dms:
            score = self.dms[dp.index]
        else:
            #tokenize data pair-wise
            s = '\t'.join([' '.join(tokens1), ' '.join(tokens2)])
            #s = itertools.chain(tokens1, tokens2, iter(' '.join(i) for i in itertools.product(tokens1, tokens2)))

            X = self.discourseModels[marker]['vectorizer'].transform([s])
            y = self.discourseModels[marker]['model'].predict(X)
            score = y[0]
            #print(s, score)
            self.dms[dp.index] = score
            
        return {self._getFeatureIndex((self.functionFeatures[self.getDiscourseModelScore] + '_' + marker, score)): 1}

    def getAllDiscourseModelScore(self, dp):
        #extract marker
        #try to find a marker
        if self.allDiscourseModel is None:
            return {}

        if not len(self.allDiscourseModel):
            print('loading model all_model')
            try:
                self.allDiscourseModel['model'] = joblib.load(os.path.join(self.modelDir,
                                                                           'all_model'))
            except Exception as e:
                print('could not open model all_model', e)
                self.allDiscourseModel = None
            print('loading model vectorizer all_model_vectorizer')
            try:
                self.allDiscourseModel['vectorizer'] = joblib.load(os.path.join(self.modelDir,
                                                                                'all_model_vectorizer'))
            except Exception as e:
                print('could not open model vectorizer all_model_vectorizer')
                self.allDiscourseModel = None

        if dp.index in self.adms:
            score = self.adms[dp.index]
        else:
            result = self._findMarker(dp.currStems)

            if result is None:
                X = self.allDiscourseModel['vectorizer'].transform([' '.join(dp.currStems)])
                y = self.allDiscourseModel['model'].predict(X)
                score = y[0]
                #print(s, score)
                self.adms[dp.index] = score
            else:
                marker, tokens1, tokens2 = result

                #tokenize data pair-wise
                s = '\t'.join([' '.join(tokens1), marker, ' '.join(tokens2)])
                #s = itertools.chain(tokens1, tokens2, iter(' '.join(i) for i in itertools.product(tokens1, tokens2)))
                X = self.allDiscourseModel['vectorizer'].transform([s])
                y = self.allDiscourseModel['model'].predict(X)
                score = y[0]
                #print(s, score)
                self.adms[dp.index] = score

        if score == 1:
            dp.numPositive +=1
        elif score == 0:
            dp.numNegative += 1
        else:
            print('score is neither positive nor negative')
            raise Exception
        
        return {self._getFeatureIndex((self.functionFeatures[self.getDiscourseModelScore], score)): 1}

    def addFeatures(self, dp, featureNames):
        features = {}
        for featureName in featureNames:
            #print(featureName)
            assert(featureName in self.validFeatures)
            ret = self.validFeatures[featureName](dp)
            features.update(ret)
            #for feature in ret:
            #    if feature not in self.featureLookup:
            #        self.featureLookup[feature] = len(self.featureLookup)
            #    features[self.featureLookup[feature]] = ret[feature]

        return features

    def addDocumentFeatures(self, dp):
        if dp.numPositive > dp.numNegative:
            feat = 'pos'
        elif dp.numNegative > dp.numPositive:
            feat = 'neg'
        else:
            feat = 'neutral'

        return {self._getFeatureIndex('doc_pol_'+feat):1}        

    def printFeatures(self, session):
        with open('features_' + session, 'w') as ff:
            for featureName in sorted(self.featureLookup, key=lambda x:self.featureLookup[x]):
                print(featureName, file=ff)
    
    def saveCache(self):
        with open(self.scoresFile, 'w') as sf:
            for index in self.scoreLookup:
                for marker in self.scoreLookup[index]:
                    print('{}\t{}\t{}'.format(index, marker, self.scoreLookup[index][marker]),
                          file=sf)
        with open(self.pscoresFile, 'w') as sf:
            for index in self.pscoreLookup:
                if index in self.pmarkerLookup:
                    marker = self.pmarkerLookup[index]
                else:
                    continue
                for polarity in self.pscoreLookup[index]:
                    print('{}\t{}\t{}\t{}'.format(index, marker, polarity, self.pscoreLookup[index][polarity]),
                          file=sf)
