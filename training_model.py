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
