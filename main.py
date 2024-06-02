import nltk
nltk.download_gui()
# Data Preprocessing
import re
import os
import csv
from nltk.stem.snowball import SnowballStemmer
import random
from nltk.classify import SklearnClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
# A command for removing all warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
#PARSING THE DOCUMENT
def extract_feature_from_doc(data):
    result = []
    corpus = []
    # The responses of the chat bot
    answers = {}
    for (text,category,answer) in data:
        features = extract_feature(text)
        corpus.append([features])
        result.append((word_feats(features), category))
        answers[category] = answer

    return (result, sum(corpus,[]), answers)
def extract_feature(text):
    # Your feature extraction logic goes here
    return text
def word_feats(words):
    return dict([(word, True) for word in words])
extract_feature_from_doc([['this is the input text from the user','category','answer to give']])
def get_content(filename):
    doc = os.path.join(filename)
    with open(doc, 'r') as content_file:
        lines = csv.reader(content_file,delimiter='|')
        data = [x for x in lines if len(x) == 3]
        return data
      filename = r"C:\Users\user\Desktop\DATA SCIENCE PROJECTS\BUILDING A CHATBOT\leaves.txt"
data = get_content(filename)
data
features_data, corpus, answers = extract_feature_from_doc(data)
print(features_data[50])
corpus
answers
#TRAINING THE MODEL
## split data into train and test sets
split_ratio = 0.8
def split_dataset(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    train_split = int(data_length * split_ratio)
    return (data[:train_split]), (data[train_split:])
training_data, test_data = split_dataset(features_data, split_ratio)
training_data
# save the data
np.save('training_data', training_data)
np.save('test_data', test_data)
#ClASSSIFICATION USING DECISION TREES
training_data = np.load('training_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy' , allow_pickle=True)
import nltk
from nltk.classify import DecisionTreeClassifier

def train_using_decision_tree(training_data, test_data):
    classifier = DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy

dtclassifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_decision_tree(training_data, test_data)
#CLASSIFICATION USING NAIVE BAYES
def train_using_naive_bayes(training_data, test_data):
    classifier = nltk.NaiveBayesClassifier.train(training_data)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy
classifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_naive_bayes(training_data, test_data)
print(training_set_accuracy)
print(test_set_accuracy)
print(len(classifier.most_informative_features()))
classifier.show_most_informative_features()
classifier.classify(({'mani': True, 'option': True, 'leav': True}))
extract_feature("hello")
word_feats(extract_feature("hello"))
input_sentence = "how many balanced leaves do I have?"
classifier.classify(word_feats(extract_feature(input_sentence)))
#TESTING
def reply(input_sentence):
    category = dtclassifier.classify(word_feats(extract_feature(input_sentence)))
    return answers[category]
reply('Hi')
reply('How many leaves have I taken?')
reply('How many annual leaves do I have left?')
reply('Thanks!')
#CONCLUSION
#Once the model has been developed using an algorithm that gives an acceptable accuracy, this model can be called using to any chatbot UI framework
!pip freeze
