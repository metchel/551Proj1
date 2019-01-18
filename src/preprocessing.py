"""
The goal of this module is to process raw json file into inputs X and outputs y 

The important function in process_features:
    takes the proj1_data.json file  as argument
    and returns X, y which are numpy arrays 
    X is 2d 12000 x 163,  y is 1d 12000 x 1
"""

import sys

def read_json_file(f):
    import os
    import json
    data = []
    with open(os.path.abspath(f)) as fp:
        data = json.load(fp)

    return data

def process_features(data, most_frequent_words):
    import numpy as np

    X = []
    y = []

    for row in data:
        y.append(row["popularity_score"])
        x_i = [int(row["children"]), float(row["controversiality"]), int(bool(row["is_root"]))] + process_text(row["text"], most_frequent_words)
        X.append(x_i)

    return np.array(X), np.array(y)

def count_words(text, k):
    all_words = {}
    for string in text:
        string = string.lower()
        words = string.split()
        for word in words:
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1
    
    top_k_words = sorted(all_words.items(), key=lambda t: t[1], reverse=True)[0:k]
    return [tup[0] for tup in top_k_words] 

def process_text(text, most_frequent_words):
    words = text.lower().split()
    word_counts = {}
    word_features = []

    for word in words:
        if word in word_counts:
            word_counts[word] +=1
        else:
            word_counts[word] = 1

    for word in most_frequent_words:
        if word in word_counts:
            word_features.append(word_counts[word])
        else:
            word_features.append(0)

    return word_features

def train_validate_test_split(X, y):
    train_X, train_y = X[:10000], y[:10000]
    validate_X, validate_y = X[10000:11000], y[10000:11000]
    test_X, test_y = X[11000:12000], y[11000:12000]

    return train_X, train_y, validate_X, validate_y, test_X, test_y

def main(args):
    data = read_json_file(args[0])
    most_frequent_words = count_words([row["text"] for row in data], 160)
    
    X, y = process_features(data, most_frequent_words)
    train_X, train_y, validate_X, validate_y, test_X, test_y = train_validate_test_split(X, y)
    print("INPUT MATRIX X: \n{}".format(X))
    print("OUTPUT VECTOR Y: \n{}".format(y))

if __name__ == "__main__": main(sys.argv[1:])
