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
