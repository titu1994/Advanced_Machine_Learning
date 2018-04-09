import time
import numpy as np
np.random.seed(0)

from proj2.utils import prepare_dataset, compute_word_char_accuracy_score
from proj2.crf_train import d_optimization_function_per_word, matricize_W, matricize_Tij, Callback
from proj2.crf_evaluate import decode_crf


# computes a cyclic learning rate schedule, where after every "stepsize_mult" number of iterations,
# the learning rate will switch from decreasing to increasing and vice versa
# This helps in speeding up convergence
def get_cyclic_triangular_lr(iteration, stepsize, current_lr, max_lr):
    stepsize = int(stepsize)
    cycle = np.floor(1 + iteration / (2 * stepsize))
    x = np.abs(iteration / stepsize - 2 * cycle + 1)
    lr = current_lr + (max_lr - current_lr) * np.maximum(0, (1 - x))
    return lr


def adam_crf(X_train, y_train, params, lambd, learning_rate, callback_fn, n_epoch=100,
             beta1=0.9, beta2=0.999, epsilon=1e-8, tol=1e-6, min_lr=1e-3, cyclic_stepsize_mult=3.):
    """ Uses the Adam optimizer with AMSGrad correction for convergence stability """
    num_words = len(X_train)
    max_lr = learning_rate

    epoch = 0
    iteration = 1

    print("Beginning training ADAM (Lambda = %f)" % (lambd))

    M = np.zeros_like(params)
    V = np.zeros_like(params)
    V_hat = np.zeros_like(params)

    np.random.seed(0)

    prev_best_loss = 1e5
    loss_update_history_counter = 0

    t1 = time.time()
    for i in range(n_epoch):
        ids = np.arange(0, num_words, dtype=int)
        np.random.shuffle(ids)

        for id in ids:
            # calculate gradient with respect to a randomly selected word
            gradient = d_optimization_function_per_word(params, X_train, y_train, id, lambd)

            # bias correction
            M = beta1 * M + (1. - beta1) * gradient
            V = beta2 * V + (1. - beta2) * np.square(gradient)

            # AMSGrad correction. It forces a non-increasing step size.
            V_hat = np.maximum(V_hat, V)

            # bias corrected first and second order moments
            m_t_hat = M / (1. - beta1 ** iteration)
            v_t_hat = V / (1. - beta2 ** iteration)

            # get cyclic learning rate
            learning_rate = get_cyclic_triangular_lr(iteration, stepsize=num_words * cyclic_stepsize_mult,
                                                     current_lr=learning_rate, max_lr=max_lr)

            params = params - (learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + epsilon)

            # update V for next update step
            V = V_hat

            iteration += 1
            learning_rate = max(learning_rate * 0.99, min_lr)

        epoch += 1

        print("Epoch %d : " % (epoch), end='')
        loss_val, avg_grad = callback_fn(params)

        # stopping criteria.  Here, if the epoch limit is reached or the gradient
        # is less than the gradient tolerance then stop
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

            max_lr /= 2.
            min_lr /= 2.

            # set a minimum learning rate of 1e-5 as AMSGrad is ineffective at such low LR
            min_lr = max(min_lr, 1e-5)
            max_lr = max(max_lr, 2e-5)

            print("Reducing minimum learning rate to %f" % (min_lr))

    print("Time taken : ", time.time() - t1)
    return params


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    LEARNING_RATES = [5e-3] # [5e-3, 2e-2, 2e-2]
    LAMBDAS = [1e-2] #[1e-2, 1e-4, 1e-6]
    TOLS = [2e-6] # [2e-6, 1e-8, 1e-8]
    MIN_LRS = [2e-3] # [5e-3, 1e-2, 1e-2]
    MAX_NUM_EPOCHS = [100] # [100, 100, 150]
    CYCLIC_STEPSIZE_MULT = [0.5] # [0.5, 0.5, 3.]

    OPTIMIZATION_NAME = "ADAM"

    FILENAME_FMT = "%s_%s.txt"

    for lambd, lr, tol, min_lr, num_epochs, stepsize_mult in zip(LAMBDAS, LEARNING_RATES, TOLS,
                                                                 MIN_LRS, MAX_NUM_EPOCHS, CYCLIC_STEPSIZE_MULT):
        params = np.zeros(129 * 26 + 26 ** 2)
        filepath = FILENAME_FMT % (OPTIMIZATION_NAME, lambd)

        callback = Callback(X_train, y_train, filepath, lambd)

        optimal_params = adam_crf(X_train, y_train, params,
                                  lambd=lambd,
                                  learning_rate=lr,
                                  callback_fn=callback.callback_fn_return_vals,
                                  n_epoch=num_epochs, beta2=0.999,
                                  tol=tol, min_lr=min_lr, cyclic_stepsize_mult=stepsize_mult)

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

