""" Contains the part of speech tagger class. """
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import csv
import string
from sklearn.metrics import accuracy_score

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

    return zip(sentences,tags_list)

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

def save_results(predicted_tags, mode):

    if mode.lower()=='dev':
        data_x = pd.read_csv('data/dev_x.csv')
        data_y = pd.read_csv('data/dev_y.csv')

    elif mode.lower()=='test':
        data_x = pd.read_csv('data/test_x.csv')
#         data_y = pd.read_csv('../results/test_y.csv')

    predicitons=[]
    print len(predicted_tags)
    for row in data_x.itertuples():
        if row[2]=='-DOCSTART-':
            predicted_tags.insert(row[1],'O')

    df = pd.DataFrame()
    df['id'] = range(len(predicted_tags))
    df['tag'] = predicted_tags
#     print "Accuracy score: " + str(accuracy_score(data_y['tag'],predicted_tags))

    if mode.lower()=='dev':
        df.to_csv('predictions.csv',index=False)
    elif mode.lower()=='test':
        df.to_csv('results/test_y.csv',index=False)

    print "Predictions saved."

def evaluate(data, model, mode='beam'):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions.

    INPUT : data, model

    OUTPUT : tokenwise accuracy, sentence accuracy

    """
    print "Evaluating."

    individual_score = 0
    sentence_score = 0
    total_word_count = 0
    final_results = []
    idx=0
    for sentence, tag_sequence in data:
        #checking number of sentences processed to gauge runtime
        idx+=1
        if idx%1000==0:
            print str(idx) + " senteces completed."
        tag_sequence = tag_sequence.split(' ')
        result = model.inference(sentence,mode)
        final_results += result
        if result==tag_sequence:
            sentence_score+=1
            individual_score+=len(result)
            total_word_count += len(result)
        else:
            for predicted, actual in zip(result,tag_sequence):
                total_word_count+=1
                if predicted==actual:
                    individual_score+=1


    print "Accuracy (tokenwise): ",float(individual_score)/total_word_count
    print "Accuracy (sentencewise): ",float(sentence_score)/len(data)

    return final_results

class POSTagger():
    def __init__(self):
        """Initializes the tagger model parameters and anything else necessary. """
        self.sentences = []
        self.tags_sentences = []
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.wordtag = {}
        self.tag_set = []
        self.word_count = {}

    def word_count_dic(self):
        word_counter = {}
        for sentence in zip(*self.train_data)[0]:
            for word in sentence.split(' '):
                self.word_count[word] = self.word_count.get(word,0) + 1
        return word_counter

    def nGramTagger(self,n):
        dic = {}
        tags = zip(*train_data)[1]
        for line in tags:
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
        dic = defaultdict(int)
        for line1,line2 in self.train_data:
            for word,tag in zip(line1.split(' '),line2.split(' ')):
                dic[(word,tag)]+=1
                if word in self.word_count and self.word_count[word]<5:
                    idx = line1.split(' ').index(word)
                    category = self.categorize_word(word, idx)
                    dic[(category,tag)]+=1
        return dic

    def KNSmoothing(self):
        probs = {}
        total = len(self.trigram)
        for tag in self.unigram.keys():
            c=0
            for item in self.trigram.keys():
                if tag==item[2]:
                    c+=1
            probs[tag]=float(c)/total

        bigram_kn_counts={}
        for key in self.bigram.keys():
            for item in self.trigram.keys():
                if key[0]==item[0] and key[1]==item[1]:
                    bigram_kn_counts[key] = bigram_kn_counts.get(key,0)+1

        return probs, bigram_kn_counts

    def get_q(self,tag_penult,tag_prev,tag_current):
        k=1
        d=0.75

        num = float(self.trigram.get((tag_penult, tag_prev, tag_current),0.0)) + k - d
        den = float(self.bigram.get((tag_penult, tag_prev), 0.0)) + k*len(self.bigram)
        lambd = float(d*self.bigram_kn_counts.get((tag_penult,tag_prev),0.0))/den
        Pcont  = self.probs.get(tag_current,0.0)

        return (num/den) + (lambd*Pcont)


    def get_e(self,word,tag,idx):
        if word not in self.word_count:
            category = self.categorize_word(word,idx)
            return float(self.wordtag[(category,tag)])/self.unigram[tag]
        else:
            return float(self.wordtag[(word,tag)])/self.unigram[tag]

    def train(self, data):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.

        """
        print "Training Model."

        self.train_data = data
        self.sentences = zip(*data)[0]
        self.tags_sentences = zip(*data)[1]

        #create dictionaries of counts of uni, bi and tri tags
        self.unigram = self.nGramTagger(1)
        self.bigram = self.nGramTagger(2)
        self.trigram = self.nGramTagger(3)
        self.word_count = self.word_count_dic()
        self.wordtag = self.wordTagger()
        self.tag_set = list(set(self.unigram.keys()))
        self.probs, self.bigram_kn_counts = self.KNSmoothing()

    def categorize_word(self,word, idx):
        if word.isdigit():
            if len(word) == 2:
                return 'twoDigitNum'
            elif len(word) == 4:
                return 'fourDigitNum'
            else:
                return 'othernum'
        elif word.isalpha():
            if word.islower():
                return 'lowercase'
            elif word.isupper():
                return 'allCaps'
            elif word[0].isupper() and word[1:].islower() and idx == 0:
                return 'firstWord'
            elif word[0].isupper() and word[1:].islower():
                return 'initCap'
            else:
                return 'other'
        else:
            without_punct = word.translate(None, string.punctuation)
            if without_punct.isdigit() and ',' in word:
                return 'containsDigitAndComma'
            elif without_punct.isdigit() and '-' in word:
                return 'containsDigitAndDash'
            elif without_punct.isdigit() and '/' in word:
                return 'containsDigitAndSlash'
            elif without_punct.isdigit() and '.' in word:
                return 'containsDigitAndPeriod'
            elif len(word) == 2 and word[0].isupper() and word[1] == '.':
                return 'capPeriod'
            elif without_punct.isalnum():
                return 'containsDigitAndAlpha'
            else:
                return 'other'

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        tag_penult = '*'
        tag_prev = '*'
        prod = 1
        idx=0
        for word, tag in zip(sequence,tags):
            q = self.get_q(tag,tag_prev,tag_penult)
            e = self.get_e(word,tag,idx)
            tag_penult = tag_prev
            tag_prev = tag
            prod *= q*e
            idx+=1

        return prod

    def possible_tags(self,k):
        if k in (-1, 0):
            return set('*')
        else:
            return self.tag_set

    def inference(self,sequence,mode):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        #Method 1: Greedy Decoding

        if mode.lower()=='greedy':
            tag_sequence = []
            tag_penult = '*'
            tag_prev = '*'
            for idx,word in enumerate(sequence.split(' ')):
                scores = []
                for tag in self.tag_set:
                    scores.append(self.get_q(tag_penult,tag_prev,tag)*self.get_e(word,tag,idx))
                final_tag = self.tag_set[np.argmax(scores)]
                tag_sequence.append(final_tag)
                tag_penult = tag_prev
                tag_prev = final_tag

            print tag_sequence
            return tag_sequence

        #Method 2: Beam Search
        elif mode.lower()=='beam':
            k=1
            sequence = sequence.split()
            best_sequences = [['*','*']]
            for idx,word in enumerate(sequence):
                scores = {}
                for item in best_sequences:
                    item = list(item)
                    tag_penult = item[-2]
                    tag_prev = item[-1]
                    for tag in self.tag_set:
                        score = self.get_q(tag_penult,tag_prev,tag)*self.get_e(word,tag,idx)
                        item.append(tag)
                        scores[tuple(item)] = score
                        item.pop()
                topk_scores = sorted(scores.items(), key=lambda x: x[1], reverse = True)
                best_sequences = list(zip(*topk_scores)[0][:k])

            best_sequence =  best_sequences[0][2:]

            return best_sequence

        #Method 3: Viterbi Algorithm
        elif mode.lower()=='viterbi':
            tag_sequence= []
            path = {}
            path['*','*'] = []
            pi_func = defaultdict(float)
            pi_func[(0, "*", '*')] = 0.0

            # v = Tag current, u = Tag previous, w = Tag penult
            sequence = sequence.split(" ")
            n = len(sequence)

            for k in range(1,n+1):
                temp_path = {}
                for u in self.possible_tags(k-1):
                    for v in self.possible_tags(k):
                        max_tag = ""
                        max_score = float("-Inf")
                        for w in self.possible_tags(k - 2):
                            score = pi_func.get((k-1, w, u),float('-Inf'))*self.get_q(w,u,v)*self.get_e(sequence[k-1],v,k-1)
                            if score > max_score:
                                max_score = score
                                max_tag = w
                        pi_func[(k, u, v)] = max_score
                        temp_path[u,v] = path[max_tag,u] + [v]
                path = temp_path

            prob,umax,vmax = max([(pi_func.get((n,u,v))*self.get_q(u,v,'.'),u,v) for u in self.possible_tags(n-1) for v in self.possible_tags(n)])

            return path[umax,vmax]




if __name__ == "__main__":
    pos_tagger = POSTagger()

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    print "Data Loaded"

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    print dev_data[0]
    predicted_dev_tags = evaluate(dev_data, pos_tagger)

    save_results(predicted_dev_tags,'dev')


    # Predict tags for the test set
    # test_predictions = []
    # for sentence in test_data:
    #     test_predictions.extend(pos_tagger.inference(sentence,mode='greedy'))

    save_results(test_predictions,'test')
