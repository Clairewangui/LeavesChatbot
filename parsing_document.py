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

