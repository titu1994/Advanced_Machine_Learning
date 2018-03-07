from subprocess import run, PIPE
import os
import matplotlib.pyplot as plt
import sys

import numpy as np
from sklearn.svm import LinearSVC

from utils import flatten_dataset, reshape_dataset, read_data_formatted
from utils import load_dataset_as_dictionary, prepare_dataset_from_dictionary, calculate_word_lengths_from_dictionary
from utils import evaluate_linearSVM, transform_dataset
from q1 import predict, get_weights

CHAR_CV_SCORES = []
WORD_CV_SCORES = []

struct_train_path = "train_struct.txt"
struct_test_path = "test_struct.txt"

def train_evaluate_linear_svm(C=1.0, transform_trainset=False, limit=None):
    X = []
    y = []

    train_data, test_data = load_dataset_as_dictionary()

    if transform_trainset:
        assert limit is not None, "If dataset is being transformed, then limit must be set"

        train_data = transform_dataset(train_data, limit)

    for key, value in train_data.items():
        X.append(value[-1])
        y.append(value[0])

    X = np.array(X)
    y = np.array(y)[:, 0]

    model = LinearSVC(C=C, max_iter=1000, verbose=10, random_state=0)
    model.fit(X, y)

    X = []
    y = []
    word_ids = []

    for key, value in test_data.items():
        X.append(value[-1])
        y.append(value[0])
        word_ids.append(value[2])

    X = np.array(X)
    y = np.array(y)[:, 0]

    y_preds = model.predict(X)
    char_acc, word_acc = evaluate_linearSVM(y, y_preds, word_ids)

    CHAR_CV_SCORES.append(char_acc)
    WORD_CV_SCORES.append(word_acc)

# evaluate CRF decoder
def evaluate_CRF(C=1.0, transform_trainset=False, limit=None):
    X = []
    y = []

    train_data, test_data = load_dataset_as_dictionary()
    
    X_temp, Wj, Tij = get_weights()
    print(len(train_data))

def plot_scores(X_range, scale='log', xlabel='C'):
    plt.plot(X_range, CHAR_CV_SCORES, label='char-level acc')
    plt.title('Character level accuracy')
    plt.legend()
    plt.xlabel(xlabel)
    if scale is not None: plt.xscale(scale)
    plt.xticks(X_range)
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(X_range, WORD_CV_SCORES, label='word-level acc')
    plt.title('Word level accuracy')
    plt.legend()
    plt.xlabel(xlabel)
    if scale is not None: plt.xscale(scale)
    plt.xticks(X_range)
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':

    ''' Linear SVM '''
    limits = [0, 500, 1000, 1500, 2000]  # [1.0, 10.0, 100.0, 1000.0]

    CHAR_CV_SCORES = []
    WORD_CV_SCORES = []

    evaluate_CRF(C=1.0, transform_trainset=False, limit=500)

    # for limit in limits:
    #     train_evaluate_linear_svm(C=1.0, transform_trainset=True, limit=limit)

    # plot_scores(limits, scale=None, xlabel='distortion count')
