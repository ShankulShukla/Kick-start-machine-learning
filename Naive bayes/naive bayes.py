import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import os

def naive_bias(vocab_count_pos, vocab_count_neg, labels, test):

    # computing the prior for each class
    priors = Counter(labels)
    class_probs = []

    # dividing the data into the classes and them compute the likelihood and the prior
    # for positive class
    prob = 1
    for word in vocab_count_pos.keys():

        if word in test:
            # multiplying each likelihood for each feature word
            prob *= vocab_count_pos[word] / float(priors[1])
        else:
            prob *= 1 - (vocab_count_pos[word] / float(priors[1]))
    posterior = prob * (priors[1] / float(len(labels)))
    class_probs.append(posterior)

    # for negative class
    prob = 1
    for word in vocab_count_neg.keys():

        if word in test:
            # multiplying each likelihood for each feature word
            prob *= vocab_count_neg[word] / float(priors[0])
        else:
            prob *= 1 - (vocab_count_neg[word] / float(priors[0]))
    posterior = prob * (priors[0] / float(len(labels)))
    class_probs.append(posterior)
    return class_probs

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

def pre_process(text):
    # first removing the signs
    text = re.sub('([^\sA-Za-z]|_)+', '', text)
    vec = text.lower().split()
    # Removing the stop words
    stop = stopwords.words('english')
    vec = [i for i in vec if i not in stop]
    # Updating words to their stemming equivalent
    stemmer = SnowballStemmer('english')
    vec = [stemmer.stem(i) for i in vec]
    return vec

def create_likelihoods(text, label, vocab_count_pos, vocab_count_neg):

    for no, sent in enumerate(text):
        vec = pre_process(sent)
        for i in vec:
            if label[no] == 1:
                vocab_count_pos[i] += 1
            else:
                vocab_count_neg[i] += 1

    return vocab_count_pos, vocab_count_neg


df = pd.read_csv(os.getcwd()+'/spam.csv',encoding='latin-1',header =None,skiprows=1)
text, label = df[1], df[0]
label = label.map({'spam': 0, 'ham': 1})
vocab_count_pos = defaultdict(lambda : 0)
vocab_count_neg = defaultdict(lambda: 0)
vocab_count_pos, vocab_count_neg = create_likelihoods(text, label, vocab_count_pos, vocab_count_neg)
test = 'Win the cash prize at the central lounge tomorrow free entry for you only !!'
print("Test e-mail:", test)
test = pre_process(test)
class_probs = naive_bias(vocab_count_pos, vocab_count_neg, label, test)
print('class probabilities - ',class_probs)
if class_probs.index(max(class_probs)) == 0:
    print('spam !!')
else:
    print('not spam !!')





