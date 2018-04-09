import time
import numpy as np
from scipy.optimize import fmin_bfgs

from proj2.utils import prepare_dataset, remove_file, compute_word_char_accuracy_score
from proj2.crf_train import objective_function, d_optimization_function, matricize_W, matricize_Tij, Callback
from proj2.crf_evaluate import decode_crf


def optimize(params, X_train, y_train, lambd, callback_fn):
    start = time.time()
    opt_params = fmin_bfgs(objective_function, params, d_optimization_function,
                           (X_train, y_train, lambd),
                           callback=callback_fn)
    print("Total time: ", end = '')
    print(time.time() - start)
    return opt_params


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")
    params = np.zeros(129 * 26 + 26 ** 2)

    LAMBDAS = [1e-2, 1e-4, 1e-6]
    OPTIMIZATION_NAME = "BFGS"

    FILENAME_FMT = "%s_%s.txt"

    for lambd in LAMBDAS:
        filepath = FILENAME_FMT % (OPTIMIZATION_NAME, lambd)

        remove_file(filepath)
        callback = Callback(X_train, y_train, filepath, lambd)

        optimal_params = optimize(params, X_train, y_train, lambd, callback.callback_fn)

        w = matricize_W(optimal_params)
        t = matricize_Tij(optimal_params)

        y_pred = decode_crf(X_train, w, t)
        word_acc, char_acc = compute_word_char_accuracy_score(y_pred, y_train)

        print("Train accuracies")
        print("Character accuracies :", char_acc)
        print("Word Accuracies :", word_acc)

        y_pred = decode_crf(X_test, w, t)
        word_acc, char_acc = compute_word_char_accuracy_score(y_pred, y_test)

        print("Test accuracies")
        print("Character accuracies :", char_acc)
        print("Word Accuracies :", word_acc)

        print()
