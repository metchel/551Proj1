import numpy as np
import time
from preprocessing import read_json_file, count_words, process_features, train_validate_test_split
from descent import sgd, momentum, adam

data = read_json_file("../data/proj1_data.json")
most_frequent_words = count_words([row["text"] for row in data], 160)
print(most_frequent_words)
X, y = process_features(data, most_frequent_words)
train_X, train_y, validate_X, validate_y, test_X, test_y = train_validate_test_split(X, y)

def least_squares(X, y):
    return np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), np.matmul(X.transpose(), y))

def train(X, y, f):
    start = time.time()
    weights = f(X, y)
    end = time.time()
    return weights, end - start

def validate(X, y, w_hat, verbose=False):
    err = []
    for i in range(len(y)):
        y_pred = (w_hat.transpose()).dot(X[i])
        err.append((y[i] - y_pred)**2)
        print("y: {}, y_pred: {}, err: {}".format(y[i], y_pred, err[i]))

    return np.mean(err)

train_y = train_y.reshape(train_y.shape[0], 1)

w0, t0 = train(train_X, train_y, least_squares)
w1, t1 = train(train_X, train_y, sgd)
w2, t2 = train(train_X, train_y, momentum)
w3, t3 = train(train_X, train_y, adam)

err = {
        "closed": (validate(validate_X, validate_y, w0), t0),
        "sgd": (validate(validate_X, validate_y, w1), t1),
        "momentum": (validate(validate_X, validate_y, w2), t2),
        "adam": (validate(validate_X, validate_y, w3), t3)
    }

print(err)
