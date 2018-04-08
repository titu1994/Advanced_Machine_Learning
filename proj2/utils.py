import numpy as np
import re
import os
from collections import defaultdict, OrderedDict

Q2_TRAIN_PATH = "data/train_sgd.txt"
Q2_TEST_PATH = "data/test_sgd.txt"

if not os.path.exists('results'):
    os.makedirs('results')


def _load_structured_svm_data(file_name):
    file = open('data/' + file_name, 'r')
    X = []
    y = []
    word_ids = []

    for line in file:
        temp = line.split()
        # get label
        label_string = ord(temp[1]) - ord('a')
        # y to 0...25 instead of 1...26
        y.append(int(label_string))

        # get word id
        word_id_string = temp[3]
        word_ids.append(int(word_id_string))

        x = np.ones(129)  # 129th value is one, the rest are overwritten by below loop
        for i, elt in enumerate(temp[5:]):
            x[i] = int(elt)

        X.append(x)

    y = np.array(y)
    word_ids = np.array(word_ids)

    return X, y, word_ids


def prepare_dataset(file_name, return_ids=False):
    # get initial output
    X, y, word_ids = _load_structured_svm_data(file_name)
    ids = {}
    X_dataset = []
    y_dataset = []
    current = 0

    X_dataset.append([])
    y_dataset.append([])

    for i in range(len(y)):
        # computes an inverse map of word id to id in the dataset
        if word_ids[i] not in ids:
            ids[word_ids[i]] = current

        X_dataset[current].append(X[i])
        y_dataset[current].append(y[i])

        if (i + 1 < len(y) and word_ids[i] != word_ids[i + 1]):
            X_dataset[current] = np.array(X_dataset[current])
            y_dataset[current] = np.array(y_dataset[current])

            X_dataset.append([])
            y_dataset.append([])

            current = current + 1

    if not return_ids:
        return X_dataset, y_dataset
    else:
        return X_dataset, y_dataset, ids


def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


def save_params(params, filename, iter):
    with open(filename, "a") as text_file:
        text_file.write("%d " % iter)
        for p in params:
            text_file.write("%f " % p)
        text_file.write("\n")
        text_file.close()


def compute_word_char_accuracy_score(y_preds, y_true):
    word_count = 0
    correct_word_count = 0
    letter_count = 0
    correct_letter_count = 0

    for y_t, y_p in zip(y_true, y_preds):
        word_count += 1
        if np.array_equal(y_t, y_p):
            correct_word_count += 1

        letter_count += len(y_p)
        correct_letter_count += np.sum(y_p == y_t)

    return correct_word_count / word_count, correct_letter_count / letter_count


def get_optimal_params(name):
    file = open('results/' + name + '.txt', 'r')
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)


if __name__ == '__main__':
    pass