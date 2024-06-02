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
