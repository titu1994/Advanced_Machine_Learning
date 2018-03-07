from subprocess import run, PIPE
import os
import matplotlib.pyplot as plt
# plt.style.use('seaborn-paper')

import numpy as np
from sklearn.svm import LinearSVC

from q1 import predict
from q2 import get_optimal_params, w_matrix, t_matrix

from utils import read_data_formatted, flatten_dataset, reshape_dataset, get_params
from utils import evaluate_structured, compute_accuracy, transform_dataset

struct_model_path = "data/model_trained.txt"
struct_test_predictions_path = "data/test_predictions.txt"

struct_train_path = "data/train_struct.txt"
struct_test_path = "data/test_struct.txt"

CHAR_CV_SCORES = []
WORD_CV_SCORES = []

def train_svm_struct_model(C=1.0):
    args = ['svm_hmm_windows/svm_hmm_learn',
            '-c', str(C),
            struct_train_path,
            struct_model_path]

    result = run(args, stdin=PIPE)


def evaluate_svm_struct_model():
    if not os.path.exists(struct_model_path):
        print("Please train the SVM-HMM model first to generate the model file.")
        exit(0)
    else:
        args = ['svm_hmm_windows/svm_hmm_classify',
                struct_test_path,
                struct_model_path,
                struct_test_predictions_path]

        result = run(args, stdin=PIPE)
        print()

        char_acc, word_acc = evaluate_structured(struct_test_path, struct_test_predictions_path)

        CHAR_CV_SCORES.append(char_acc)
        WORD_CV_SCORES.append(word_acc)


def train_evaluate_linear_svm(C=1.0, transform_trainset=False, limit=None):
    X = []
    y = []

    X_train, Y_train = read_data_formatted('train_struct.txt')
    X_test, Y_test = read_data_formatted('test_struct.txt'3)

    word_ids = []

    for i in range(len(X_train)):
        word_ids.append(len(X_train[i]))

    x_train = flatten_dataset(X_train)
    y_train = flatten_dataset(Y_train)

    x_test = flatten_dataset(X_test)

    if transform_trainset:
        assert limit is not None, "If dataset is being transformed, then limit must be set"

        train_data = transform_dataset(X_train, limit, word_ids)

    model = LinearSVC(C=C, max_iter=1000, verbose=10, random_state=0)
    model.fit(x_train, y_train)

    y_preds = model.predict(x_test)

    y_preds = reshape_dataset(y_preds, Y_test)  # Y_test represents the word indices
    word_acc, char_acc = compute_accuracy(y_preds, Y_test)

    CHAR_CV_SCORES.append(char_acc)
    WORD_CV_SCORES.append(word_acc)


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
    """ Question 3 """

    ''' Structured SVM '''
    # # Used for grid search and plotting
    # Cs = [1.0, 10.0, 100.0, 1000.0] # [1.0, 10.0, 100.0, 1000.0]
    #
    # # Used for getting maximum test scores at word and character level
    # #Cs = [1000.0]
    #
    # CHAR_CV_SCORES.clear()
    # WORD_CV_SCORES.clear()
    #
    # for C in Cs:
    #     train_svm_struct_model(C=C)
    #     evaluate_svm_struct_model()
    #
    # plot_scores(Cs)
    #
    ''' Linear Multi-Class SVM '''
    # # Used for grid search and plotting
    # Cs = [1.0, 10.0, 100.0, 1000.0]
    #
    # # Used for getting maximum test scores at word and character level
    # # Cs = [1.0]
    #
    # CHAR_CV_SCORES.clear()
    # WORD_CV_SCORES.clear()
    #
    # for C in Cs:
    #     train_evaluate_linear_svm(C=C)
    #
    # plot_scores(Cs)

    ''' CRF '''

    X_train, Y_train = read_data_formatted('train_struct.txt')
    X_test, Y_test = read_data_formatted('test_struct.txt')
    params = get_params()

    x_test = flatten_dataset(X_test)
    y_test = flatten_dataset(Y_test)

    # Run optimization, should take 3+ hours so commented out.
    '''
    cvals = [1, 10, 100, 1000]
    for elt in cvals:
        p2b.optimize(params, X_train, y_train, elt, 'solution' + str(elt))
        print("done with" + str(elt))
    '''

    # check accuracy
    Cs = [1, 10, 100, 1000]

    CHAR_CV_SCORES.clear()
    WORD_CV_SCORES.clear()

    for C in Cs:
        print("\nComputing predictions for C = %d" % (C))
        params = get_optimal_params('solution' + str(C))
        w = w_matrix(params)
        t = t_matrix(params)
        prediction = predict(x_test, w, t)

        prediction = reshape_dataset(prediction, Y_test)  # y_test is for getting word ids

        word_acc, char_acc = compute_accuracy(prediction, Y_test)

        CHAR_CV_SCORES.append(char_acc)
        WORD_CV_SCORES.append(word_acc)

    plot_scores(Cs)



