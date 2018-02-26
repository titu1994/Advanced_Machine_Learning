from subprocess import run, PIPE
import os

from utils import evaluate_structured

struct_model_path = "data/model_trained.txt"
struct_test_predictions_path = "data/test_predictions.txt"

struct_train_path = "data/train_struct.txt"
struct_test_path = "data/test_struct.txt"


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

        evaluate_structured(struct_test_path, struct_test_predictions_path)



if __name__ == '__main__':
    pass
    # train_svm_struct_model(C=1.0)

    evaluate_svm_struct_model()
