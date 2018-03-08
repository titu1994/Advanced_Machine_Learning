from subprocess import run, PIPE
import os
import matplotlib.pyplot as plt
# plt.style.use('seaborn-paper')

import numpy as np
from sklearn.svm import LinearSVC

from q2.q1 import decode_crf
from q2.q2 import get_optimal_params, w_matrix, t_matrix

from q2.utils import read_data_formatted, flatten_dataset, reshape_dataset, get_params
from q2.utils import evaluate_structured, compute_accuracy, transform_dataset

struct_model_path = "data/model_trained.txt"
struct_test_predictions_path = "data/test_predictions.txt"

struct_train_path = "data/train_struct.txt"
struct_test_path = "data/test_struct.txt"

# used for plotting
CHAR_CV_SCORES = []
WORD_CV_SCORES = []

# call the structured svm application
def train_svm_struct_model(C=1.0):
    # CLI arguments
    args = ['svm_hmm_windows/svm_hmm_learn',
            '-c', str(C),
            struct_train_path,
            struct_model_path]

    result = run(args, stdin=PIPE)


# use the generated model file to predict labels
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

        # evaluate svm-hmm results in its specific format
        char_acc, word_acc = evaluate_structured(struct_test_path, struct_test_predictions_path)

        CHAR_CV_SCORES.append(char_acc)
        WORD_CV_SCORES.append(word_acc)


# train and evaluate a linear SVM
def train_evaluate_linear_svm(C=1.0):
    X_train, Y_train = read_data_formatted('train_struct.txt')
    X_test, Y_test = read_data_formatted('test_struct.txt')

    # linear svm takes data of shape (N, 128) as input
    x_train = flatten_dataset(X_train)
    y_train = flatten_dataset(Y_train)
    x_test = flatten_dataset(X_test)

    # train the model
    model = LinearSVC(C=C, verbose=10, random_state=0)
    model.fit(x_train, y_train)

    # evaluate the model
    y_preds = model.predict(x_test)

    # reshape the predictions into a list of words
    y_preds = reshape_dataset(y_preds, Y_test)  # Y_test represents the word indices

    # compute accuracy
    word_acc, char_acc = compute_accuracy(y_preds, Y_test)

    CHAR_CV_SCORES.append(char_acc)
    WORD_CV_SCORES.append(word_acc)


# plot the scores
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
    # Used for grid search and plotting
    Cs = [1.0, 10.0, 100.0, 1000.0] # [1.0, 10.0, 100.0, 1000.0]

    # Used for getting maximum test scores at word and character level
    #Cs = [1000.0]

    CHAR_CV_SCORES.clear()
    WORD_CV_SCORES.clear()

    for C in Cs:
        train_svm_struct_model(C=C)
        evaluate_svm_struct_model()

    plot_scores(Cs)

    ''' Linear Multi-Class SVM '''
    # Used for grid search and plotting
    Cs = [1.0, 10.0, 100.0, 1000.0]

    # Used for getting maximum test scores at word and character level
    # Cs = [1.0]

    CHAR_CV_SCORES.clear()
    WORD_CV_SCORES.clear()

    for C in Cs:
        train_evaluate_linear_svm(C=C)

    plot_scores(Cs)

    ''' CRF '''

    X_train, Y_train = read_data_formatted('train_struct.txt')
    X_test, Y_test = read_data_formatted('test_struct.txt')
    params = get_params()

    x_test = flatten_dataset(X_test)
    y_test = flatten_dataset(Y_test)

    # Run optimization, should take 3.5+ hours so commented out.
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
        # get pretrained optimal params
        params = get_optimal_params('solution' + str(C))
        w = w_matrix(params)
        t = t_matrix(params)

        # get predictions
        prediction = decode_crf(x_test, w, t)

        # reshape into a list of words
        prediction = reshape_dataset(prediction, Y_test)  # y_test is for getting word ids

        # compute accuracy
        word_acc, char_acc = compute_accuracy(prediction, Y_test)

        CHAR_CV_SCORES.append(char_acc)
        WORD_CV_SCORES.append(word_acc)

    plot_scores(Cs)



