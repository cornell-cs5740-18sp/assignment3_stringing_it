""" Contains the part of speech tagger class. """

def make_sentences(tokens,tags):
    """
    Function converts list of words into sentences with sentences of corresponding tags

    INPUT : Dataframe of tokens, Dataframe of tags

    OUTPUT : Zip of list of sentences, list of tags sentences
    """
    data = tokens.join(tags, on="id", how = "inner", rsuffix = "_tag").drop("id_tag",axis=1)
    sentences = []
    tags_list = []
    temp_tokens = []
    temp_tags = []
    for row in data.itertuples():
        word = row[2]
        tag = row[3]
        if word!='-DOCSTART-' and word!='.':
            temp_tokens.append(word)
            temp_tags.append(tag)
        if word=='.':
            sentences.append(' '.join(temp_tokens) + ' .')
            tags_list.append(' '.join(temp_tags) + ' .')
            temp_tokens = []
            temp_tags = []

    return zip(sentences,tags)

def load_data(sentence_file, tag_file=None):
    """Loads data from two files: one containing sentences and one containing tags.

    tag_file is optional, so this function can be used to load the test data.

    Suggested to split the data by the document-start symbol.

    """
    tokens = pd.read_csv(sentence_file)
    if tag_file:
        tags = pd.read_csv(tag_file)
    else:
        #dummy tags for test file
        tags = pd.DataFrame()
        tags['id'] = range(len(tokens))
        tags['tag'] = ['NNP']*len(tokens)

    return make_sentences(tokens,tags)

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.

    """
    pass

class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.sentences = []
        self.tags_sentences = []
        self.unitag = {}
        self.bitag = {}
        self.tritag = {}
        self.wordtag = {}
        self.tag_set = set()

    def nGramTagger(self,n):
        """
        Computes n-gram tag count dictionary for estimating transition probabilitites
        INPUT : int (n)
        OUTPUT Dict (n-gram tag count dictionary)
        """
        dic = {}
        for line in self.tags_sentences:
            line = line.split(' ')
            line = ['*']*n + line
            for i in range(n,len(line)):
                if n==1:
                    item = line[i]
                else:
                    item = tuple(line[i-n:i])
                if item in dic:
                    dic[item]+=1
                else:
                    dic[item]=1
        return dic

    def wordTagger(self):
        """
        Computes word,tag count dictionary for estimating emission probabilitites
        INPUT : None
        OUTPUT : Dict ((word,tag) count dictionary)
        """

        dic = defaultdict(int)
        for line1,line2 in zip(self.sentences,self.tags_sentences):
            for word,tag in zip(line1.split(' '),line2.split(' ')):
                dic[(word,tag)]+=1
        return dic

    def get_q(self,tag_penult,tag_last,tag_current):
        """
        Computes transition probabilitites for trigram tagger
        INPUT : (current tag, previous tag, penultimate tag) || (string, string, string)
        OUTPUT : (transition probabilty) || (float)
        """

        return float(self.tritag[(tag_penult, tag_last, tag_current)])/self.bitag[(tag_penult, tag_last)]

    def get_e(self,word,tag):
        """
        Computes emission probabilitites for trigram tagger
        INPUT : (word, tag) || (string, string)
        OUTPUT : (emission probabilty) || (float)
        """

        return float(self.wordtag[(word,tag)])/self.unitag[tag]

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.

        """
        self.sentences = zip(*data)[0]
        self.tags_sentences = zip(*data)[1]

        #create dictionaries of counts of uni, bi and tri tags
        self.unitag = self.nGramTagger(1)
        self.bitag = self.nGramTagger(2)
        self.tritag = self.nGramTagger(3)

        self.wordtag = self.wordTagger()
        self.tag_set = set(self.unitag.keys())


    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        tag_penult = '*'
        tag_prev = '*'
        prod = 1
        for word, tag in zip(sequence,tags):
            q = self.get_q(tag,tag_prev,tag_penult)
            e = self.get_e(word,tag)
            tag_penult = tag_prev
            tag_prev = tagger
            prod *= q*e

        return prod

    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """



if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence)

    # Write them to a file to update the leaderboard
    # TODO
