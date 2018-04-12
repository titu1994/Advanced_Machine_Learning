from subprocess import run, PIPE
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

import numpy as np
from sklearn.svm import LinearSVC

from data_loader import load_Q2_data
from utils import evaluate_structured, evaluate_crf, transform_dataset

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

    train_data, test_data = load_Q2_data()

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
    char_acc, word_acc = evaluate_crf(y, y_preds, word_ids)

    CHAR_CV_SCORES.append(char_acc)
    WORD_CV_SCORES.append(word_acc)


def plot_scores(X_range, scale='log', xlabel='C'):
    plt.plot(X_range, CHAR_CV_SCORES, label='char-level result')
    plt.title('Character level accuracy')
    plt.legend()
    plt.xlabel(xlabel)
    if scale is not None: plt.xscale(scale)
    plt.xticks(X_range)
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(X_range, WORD_CV_SCORES, label='word-level result')
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
    # Used for grid search and plotting
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

    ''' Linear Multi-Class SVM '''
    # Used for grid search and plotting
    # NOTE: The values for C are inverted, as Scikit-Learn uses
    # an inverse C value. Larger values in liblinear = smaller values
    # in scikit-learn.
    # Cs = [1e-3, 1e-2, 1e-1, 1.0]  # [1.0, 10.0, 100.0, 1000.0]
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


    """ Question 4 """
    # Used for grid search and plotting
    # NOTE: The values for C are inverted, as Scikit-Learn uses
    # an inverse C value. Larger values in liblinear = smaller values
    # in scikit-learn.
    limits = [0, 500, 1000, 1500, 2000]  # [1.0, 10.0, 100.0, 1000.0]

    # Used for getting maximum test scores at word and character level
    # Cs = [1.0]

    CHAR_CV_SCORES.clear()
    WORD_CV_SCORES.clear()

    for limit in limits:
        train_evaluate_linear_svm(C=1.0, transform_trainset=True, limit=limit)

    plot_scores(limits, scale=None, xlabel='distortion count')


