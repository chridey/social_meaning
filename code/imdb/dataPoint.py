from nltk.stem import SnowballStemmer

class DataPoint:
    sn = SnowballStemmer("english")
    
    def __init__(self, dataDict):
        self._dataDict = dataDict
        self.numPositive = 0
        self.numNegative = 0
        
    @property
    def currWords(self):
        return self._dataDict['curr_words']

    @property
    def prevWords(self):
        return self._dataDict['prev_words']

    @property
    def currStems(self):
        if 'curr_stems' not in self._dataDict:
            self._dataDict['curr_stems'] = [self.sn.stem(t) for t in self._dataDict['curr_words']]
        return self._dataDict['curr_stems']

    @property
    def prevStems(self):
        if 'prev_stems' not in self._dataDict:
            if 'prev_words' in self._dataDict and self._dataDict['prev_words'] is not None:
                self._dataDict['prev_stems'] = [self.sn.stem(t) for t in self._dataDict['prev_words']]
            else:
                self._dataDict['prev_stems'] = None
        return self._dataDict['prev_stems']

    @property
    def relIndex(self):
        return 1.0*self._dataDict['curr_index']/self._dataDict['global_info']['num_sentences']

    @property
    def relLength(self):
        return 1.0*len(self.currWords)/self._dataDict['global_info']['max_sentence_length']
    
    @property
    def index(self):
        return '{}_{}'.format(self._dataDict['filename'],
                              self._dataDict['curr_index'])

    @property
    def currSubj(self):
        return self._dataDict['curr_subj']
    
    @property
    def prevSubj(self):
        return self._dataDict['prev_subj']
