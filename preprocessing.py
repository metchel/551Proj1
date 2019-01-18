"""
The goal of this module is to process raw json file into inputs X and outputs y 

The important function in process_features:
    takes the proj1_data.json file  as argument
    and returns X, y which are numpy arrays 
    X is 2d 12000 x 163,  y is 1d 12000 x 1
"""

import sys
import os
import json
import numpy as np

def process_features(f):
    X = []
    y = []

    with open(os.path.abspath(f)) as fp:
        data = json.load(fp)
        most_frequent_words = count_words([row["text"] for row in data], 160)
        
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

def main(args):
    X, y = process_features(args[0])
    print("INPUT MATRIX X: \n{}".format(X))
    print("OUTPUT VECTOR Y: \n{}".format(y))

if __name__ == "__main__": main(sys.argv[1:])
