import numpy as np
from preprocessing import read_json_file, count_words, process_features, train_validate_test_split


data = read_json_file("../data/proj1_data.json")
most_frequent_words = count_words([row["text"] for row in data], 160)
print(most_frequent_words)
X, y = process_features(data, most_frequent_words)
train_X, train_y, validate_X, validate_y, test_X, test_y = train_validate_test_split(X, y)

def least_squares(X, y):
    return np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y))
print(least_squares(train_X, train_y))



