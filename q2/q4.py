import matplotlib.pyplot as plt

import numpy as np
from sklearn.svm import LinearSVC

from utils import flatten_dataset, reshape_dataset, read_data_formatted, compute_accuracy, get_params
from utils import load_dataset_as_dictionary, prepare_dataset_from_dictionary, calculate_word_lengths_from_dictionary
from utils import evaluate_linearSVM, transform_dataset, transform_crf_dataset

from q1 import predict, get_weights
from q2 import get_optimal_params, w_matrix, t_matrix, optimize

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
    limits = [ 2000 ]

    CHAR_CV_SCORES = []
    WORD_CV_SCORES = []

    ''' linear SVM '''
    for limit in limits:
        train_evaluate_linear_svm(C=1.0, transform_trainset=True, limit=limit)

    plot_scores(limits, scale=None, xlabel='distortion count')

    ''' CRF '''
    CHAR_CV_SCORES = []
    WORD_CV_SCORES = []

    X_test, Y_test = read_data_formatted('test_struct.txt')

    print("Computing scores for best model with no distortion")
    # compute the test scores for best model first
    x_test = flatten_dataset(X_test)
    params = get_optimal_params('solution')
    w = w_matrix(params)
    t = t_matrix(params)

    y_preds = predict(x_test, w, t)
    y_preds = reshape_dataset(y_preds, Y_test)

    word_acc, char_acc = compute_accuracy(y_preds, Y_test)
    CHAR_CV_SCORES.append(char_acc)
    WORD_CV_SCORES.append(word_acc)

    for limit in limits:
        print("Beginning distortaion of first %d ids" % (limit))

        X_train, Y_train = read_data_formatted('train_struct_%d.txt' % (limit))
        # optimize(params, X_train, Y_train, C=1500, name='solution_%d_distortion' % limit)

        params = get_optimal_params('solution_%d_distortion' % limit)
        w = w_matrix(params)
        t = t_matrix(params)

        prediction = predict(x_test, w, t)
        prediction = reshape_dataset(prediction, Y_test)  # y_test is for getting word ids

        word_acc, char_acc = compute_accuracy(prediction, Y_test)

        CHAR_CV_SCORES.append(char_acc)
        WORD_CV_SCORES.append(word_acc)

    limits = [0] + limits
    plot_scores(limits, scale=None, xlabel='distortion count')



