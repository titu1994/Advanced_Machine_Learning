import time
import numpy as np
np.random.seed(0)

from utils import *
from crf_train import *
from crf_evaluate import decode_crf

"""

NOTE : THIS CODE DOES NOT CURRENTLY WORK

"""
def adam_crf_mcmc(X_train, y_train, params, lambd, learning_rate, callback_fn, n_epoch=100, num_samples=3,
                  beta1=0.9, beta2=0.999, epsilon=1e-8):
    epoch = 0
    iteration = 1

    print("Beginning training ADAM (Lambda = %f)" % (lambd))

    M = np.zeros_like(params)
    V = np.zeros_like(params)
    V_hat = np.zeros_like(params)

    t1 = time.time()
    while (True):

        id = np.random.randint(0, len(X_train), dtype=int)
        # calculate gradient with respect to a randomly selected word
        gradient = grad_func_word_mcmc(params, X_train, y_train, id, lambd, num_samples)

        # biased moments
        M = beta1 * M + (1. - beta1) * gradient
        V = beta2 * V + (1. - beta2) * np.square(gradient)

        V_hat = np.maximum(V_hat, V)

        # bias corrected first and second order moments
        m_t_hat = M / (1. - beta1 ** iteration)
        v_t_hat = V / (1. - beta2 ** iteration)

        params = params - (learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + epsilon)

        V = V_hat

        # stopping criteria.  Here, if the epoch limit is reached or the gradient
        # is less than the gradient tolerance then stop
        iteration += 1

        if (iteration % (len(X_train) * num_samples) == 0):
            epoch += 1

            learning_rate = max(learning_rate * 0.9, 5e-4)

            print("Epoch %d : " % (epoch), end='')
            loss_val, avg_grad = callback_fn(params)

            if avg_grad < 5e-6:
                break

            if (epoch >= n_epoch):
                print("Epoch limit")
                break

    print("Time taken : ", time.time() - t1)

    return params
gradient = np.zeros((26, 129))

def resolve_category(category):
    if category == 0:  # low
        return 30
    elif category == 1:  # medium
        return 60
    else:  # high
        return 100


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    CATEGORY = 0  # 0 = low, 1 = med, 2 = high

    LEARNING_RATES = [1e-2, 1e-2, 1e-2]
    LAMBDAS = [1e-2, 1e-4, 1e-6]
    OPTIMIZATION_NAME = "ADAM_MCMC"

    FILENAME_FMT = "%s_%s.txt"
    NUM_SAMPLES = resolve_category(CATEGORY)


    for lambd, lr in zip(LAMBDAS, LEARNING_RATES):
        params = np.zeros(129 * 26 + 26 ** 2)
        filepath = FILENAME_FMT % (OPTIMIZATION_NAME, lambd)

        remove_file(filepath)
        callback = Callback(X_train, y_train, filepath, lambd)

        optimal_params = adam_crf_mcmc(X_train, y_train, params, lambd,
                                       lr, callback.callback_fn_return_vals,
                                       n_epoch=100, num_samples=3)

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

