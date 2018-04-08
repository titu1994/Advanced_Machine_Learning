import time
import numpy as np
np.random.seed(0)

from proj2.utils import *
from proj2.crf_train import *
from proj2.crf_evaluate import decode_crf


# def sgd_crf(X_train, y_train, params, lambd, learning_rate, callback_fn, n_epoch=100):
#     epoch = 0
#     iteration = 0
#
#     print("Beginning training SGD (Lambda = %f)" % (lambd))
#
#     t1 = time.time()
#     while (True):
#
#         id = np.random.randint(0, len(X_train), dtype=int)
#
#         # calculate gradient with respect to a randomly selected word
#         gradient = grad_func_word(params, X_train, y_train, id, lambd)
#
#         params = params - learning_rate * gradient
#
#         # stopping criteria.  Here, if the epoch limit is reached or the gradient
#         # is less than the gradient tolerance then stop
#         iteration += 1
#
#         if (iteration % len(X_train) == 0):
#             epoch += 1
#
#             learning_rate = max(learning_rate * 0.9, 5e-4)
#
#             print("Epoch %d : " % (epoch), end='')
#             avg_grad = callback_fn(params)
#
#             if avg_grad < 1e-6:
#                 break
#
#             if (epoch >= n_epoch):
#                 print("Epoch limit")
#                 break
#
#     print("Time taken : ", time.time() - t1)
#
#     return params


def sgd_crf(X_train, y_train, params, lambd, learning_rate, callback_fn, n_epoch=100):
    epoch = 0
    iteration = 0

    print("Beginning training SGD (Lambda = %f)" % (lambd))

    running_params = np.zeros_like(params)

    GAMMA = 0.9

    t1 = time.time()
    while (True):

        id = np.random.randint(0, len(X_train), dtype=int)

        # calculate gradient with respect to a randomly selected word
        gradient = grad_func_word(params, X_train, y_train, id, lambd)

        running_params = GAMMA * running_params + (1 - GAMMA) * gradient

        params = params - learning_rate * running_params

        # stopping criteria.  Here, if the epoch limit is reached or the gradient
        # is less than the gradient tolerance then stop
        iteration += 1

        if (iteration % len(X_train) == 0):
            epoch += 1

            learning_rate = max(learning_rate * 0.9, 5e-4)

            print("Epoch %d : " % (epoch), end='')
            avg_grad = callback_fn(params)

            if avg_grad < 5e-6:
                break

            if (epoch >= n_epoch):
                print("Epoch limit")
                break

    print("Time taken : ", time.time() - t1)

    return params


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    LEARNING_RATES = [1e-2, 1e-2, 1e-2]
    LAMBDAS = [1e-2, 1e-4, 1e-6]
    OPTIMIZATION_NAME = "SGD"

    FILENAME_FMT = "%s_%s.txt"

    for lambd, lr in zip(LAMBDAS, LEARNING_RATES):
        params = np.zeros(129 * 26 + 26 ** 2)
        filepath = FILENAME_FMT % (OPTIMIZATION_NAME, lambd)

        callback = Callback(X_train, y_train, filepath, lambd)

        optimal_params = sgd_crf(X_train, y_train, params, lambd,
                                 lr, callback.callback_fn_return_avg_grad,
                                 n_epoch=100)

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

