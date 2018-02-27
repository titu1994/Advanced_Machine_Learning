from subprocess import run, PIPE
import os

import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

from Project1.data_loader import load_Q2_data
from utils import evaluate_structured

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


if __name__ == '__main__':

    # Used for grid search and plotting
    Cs = [1.0, 10.0, 100.0, 1000.0] # [1.0, 10.0, 100.0, 1000.0]

    # Used for getting maximum test scores at word and character level
    #Cs = [1000.0]

    for C in Cs:
        train_svm_struct_model(C=C)
        evaluate_svm_struct_model()

    plt.plot(Cs, CHAR_CV_SCORES, label='char-level acc')
    plt.title('Character level accuracy')
    plt.legend()
    plt.xlabel('C')
    plt.xscale('log')
    plt.xticks(Cs)
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(Cs, WORD_CV_SCORES, label='word-level acc')
    plt.title('Word level accuracy')
    plt.legend()
    plt.xlabel('C')
    plt.xscale('log')
    plt.xticks(Cs)
    plt.ylabel('accuracy')
    plt.show()


