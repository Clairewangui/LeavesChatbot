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
