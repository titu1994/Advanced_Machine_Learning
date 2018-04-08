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


def save_losses(loss, filename, iter):
    with open(filename, "a") as f:
        f.write("%d %f\n" % (iter, loss))


def save_params(params, filename, iter):
    with open(filename, "a") as f:
        f.write("%d " % iter)
        for p in params:
            f.write("%f " % p)
        f.write("\n")


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


# helper methods for tensorflow
def load_dataset_as_dictionary():
    train_data = OrderedDict()

    with open(Q2_TRAIN_PATH, 'r') as f:
        lines = f.readlines()

    for l in lines:
        letter = re.findall(r'[a-z]', l)
        l = re.findall(r'\d+', l)
        letter_id = l[0]
        next_id = l[1]
        word_id = l[2]
        pos = l[3]
        p_ij = np.ones((129,), dtype=np.float32)
        p_ij[:128] = np.array(l[4:], dtype=np.float32)
        train_data[letter_id] = [letter, next_id, word_id, pos, p_ij]

    test_data = OrderedDict()

    with open(Q2_TEST_PATH, 'r') as f:
        lines = f.readlines()

    for l in lines:
        letter = re.findall(r'[a-z]', l)
        l = re.findall(r'\d+', l)
        letter_id = l[0]
        next_id = l[1]
        word_id = l[2]
        pos = l[3]
        p_ij = np.ones((129,), dtype=np.float32)
        p_ij[:128] = np.array(l[4:], dtype=np.float32)
        # store letter in dictionary as letter_id -> letter, next_id, word_id, position, pixel_values
        test_data[letter_id] = [letter, next_id, word_id, pos, p_ij]

    return train_data, test_data


# helper methods for tensorflow
def calculate_word_lengths_from_dictionary(dataset):
    word_list = []
    prev_word = -100

    for i, (key, value) in enumerate(dataset.items()):
        letter, next_id, word_id, pos, p_ij = value

        if word_id == prev_word:
            word_list[-1] += 1
        else:
            word_list.append(1)
            prev_word = word_id

    return word_list


# helper methods for tensorflow
def prepare_dataset_from_dictionary(dataset, word_length_list):
    max_length = max(word_length_list)
    num_samples = len(word_length_list)

    X = np.zeros((num_samples, max_length, 129), dtype='float32')
    y = np.zeros((num_samples, max_length), dtype='int32')

    dataset_pointer = 0
    dataset_values = list(dataset.values())

    for i in range(num_samples):
        num_words = word_length_list[i]

        for j in range(num_words):
            letter, next_id, word_id, pos, p_ij = dataset_values[j + dataset_pointer]
            X[i, j, :] = p_ij

            letter_to_int = int(ord(letter[0]) - ord('a'))
            y[i, j] = letter_to_int

        dataset_pointer += num_words

    return X, y


if __name__ == '__main__':
    pass