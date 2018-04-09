import time
import numpy as np
np.random.seed(0)

from utils import prepare_dataset, compute_word_char_accuracy_score
from crf_train import grad_func_word, matricize_W, matricize_Tij, Callback
from crf_evaluate import decode_crf


def sgd_crf(X_train, y_train, params, lambd, learning_rate, callback_fn, n_epoch=100, tol=1e-6,
            nesterov=False, min_lr=5e-4):
    epoch = 0
    iteration = 0

    print("Beginning training SGD (Lambda = %f)" % (lambd))

    np.random.seed(0)

    prev_best_loss = 1e5
    loss_update_history_counter = 0

    t1 = time.time()
    for i in range(n_epoch):
        ids = np.arange(0, len(X_train), dtype=int)
        np.random.shuffle(ids)

        moving_params = np.zeros_like(params)
        GAMMA = 0.9

        for id in ids:
            if nesterov:
                # apply nesterov update first
                nesterov_params = params - learning_rate * moving_params

                # calculate the gradient with respect to a randomly selected word using nesterov params
                gradient = grad_func_word(nesterov_params, X_train, y_train, id, lambd)

                # apply momentum update
                moving_params = GAMMA * moving_params + (1. - GAMMA) * gradient

                # parameter update
                params = params - learning_rate * moving_params

            else:
                # calculate gradient with respect to a randomly selected word
                gradient = grad_func_word(params, X_train, y_train, id, lambd)

                # apply momentum update
                moving_params = GAMMA * moving_params + (1. - GAMMA) * gradient

                # parameter update
                params = params - learning_rate * moving_params

            # stopping criteria.  Here, if the epoch limit is reached or the gradient
            # is less than the gradient tolerance then stop
            iteration += 1
            learning_rate = max(learning_rate * 0.99, min_lr)

        epoch += 1
        print("Epoch %d : " % (epoch), end='')
        loss_val, avg_grad = callback_fn(params)

        if avg_grad < tol:
            break

        # check if loss decreased or not
        if prev_best_loss > loss_val:
            prev_best_loss = loss_val
            loss_update_history_counter = 0  # reset the flag counter
        else:
            loss_update_history_counter += 1  # increment the flag counter

        # if the flag counter is more than 3, reduce the minimum learning rate by half
        # this causes the current learning rate to drop more
        if loss_update_history_counter >= 3:
            loss_update_history_counter = 0
            min_lr /= 2.

            print("Reducing minimum learning rate to %f" % (min_lr))

    print("Time taken : ", time.time() - t1)

    return params


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    LEARNING_RATES = [0.1, 0.2, 0.1]
    LAMBDAS = [1e-2, 1e-4, 1e-6]
    TOLS = [2e-6, 1e-8, 1e-8]
    MIN_LRS = [5e-3, 2e-2, 2e-2]
    MAX_NUM_EPOCHS = [100, 100, 150]

    OPTIMIZATION_NAME = "SGD"

    FILENAME_FMT = "%s_%s.txt"

    for lambd, lr, tol, min_lr, num_epochs in zip(LAMBDAS, LEARNING_RATES, TOLS, MIN_LRS, MAX_NUM_EPOCHS):
        params = np.zeros(129 * 26 + 26 ** 2)
        filepath = FILENAME_FMT % (OPTIMIZATION_NAME, lambd)

        callback = Callback(X_train, y_train, filepath, lambd)

        optimal_params = sgd_crf(X_train, y_train, params,
                                 lambd=lambd,
                                 learning_rate=lr,
                                 callback_fn=callback.callback_fn_return_vals,
                                 n_epoch=num_epochs, tol=tol, nesterov=True, min_lr=min_lr)

        w = matricize_W(optimal_params)
        t = matricize_Tij(optimal_params)

        y_pred = decode_crf(X_train, w, t)
        word_acc, char_acc = compute_word_char_accuracy_score(y_pred, y_train)

        print("Train accuracies")
        print("Character accuracies :", char_acc)
        print("Word Accuracies :", word_acc)
        print()

        y_pred = decode_crf(X_test, w, t)
        word_acc, char_acc = compute_word_char_accuracy_score(y_pred, y_test)

        print("Test accuracies")
        print("Character accuracies :", char_acc)
        print("Word Accuracies :", word_acc)
        print()

