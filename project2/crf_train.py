import numpy as np
import time
from scipy.optimize import check_grad, fmin_bfgs

from project2.utils import prepare_structured_dataset, load_model_params, load_model_params_zeros, compute_word_char_accuracy_score
from project2.crf_evaluate import decode_crf

def compute_forward_message(x, w, t):
    w_x = np.dot(x,w.T)
    num_words = len(w_x)
    M = np.zeros((num_words, 26))

    # iterate through all characters in each word
    for i in range(1, num_words):
        alpha = M[i - 1] + t.transpose()
        alpha_max = np.max(alpha, axis=1)
        # prepare V - V.max()
        alpha = (alpha.transpose() - alpha_max).transpose()
        M[i] = alpha_max + np.log(np.sum(np.exp(alpha + w_x[i - 1]), axis=1))

    return M


def compute_backward_message(x, w, t):
    # get the index of the final letter of the word
    w_x = np.dot(x,w.T)
    fin_index = len(w_x) - 1
    M = np.zeros((len(w_x), 26))

    for i in range(fin_index - 1, -1, -1):
        beta = M[i + 1] + t
        beta_max = np.max(beta, axis=1)
        # prepare V - V.max()
        beta = (beta.transpose() - beta_max).transpose()
        M[i] = beta_max + np.log(np.sum(np.exp(beta + w_x[i + 1]), axis=1))

    return M


def compute_numerator(y, x, w, t):
    w_x = np.dot(x,w.T)
    sum_ = 0
    # for every word
    for i in range(len(w_x)):
        sum_ += w_x[i][y[i]]

        if (i > 0):
            # t stored as T{current, prev}
            sum_ += t[y[i - 1]][y[i]]

    return np.exp(sum_)


# def compute_denominator(alpha, x, w):
#     # forward propagate to the end of the word and return the sum
#     return np.log(np.sum(np.exp(alpha[-1] + np.dot(x,w.T)[-1])))


def compute_denominator(alpha, x, w):
    # forward propagate to the end of the word and return the sum
    intermediate = alpha[-1] + np.dot(x,w.T)[-1]
    max_inter = np.max(intermediate)
    intermediate -= max_inter
    out = max_inter + np.log(np.sum(np.exp(intermediate)))
    return out


# convert W into a matrix from its flattened parameters

def matricize_W(params):
    w = params[:26 * 129]
    w = w.reshape((26, 129))

    #for i in range(26):
    #    w[i] = params[129 * i: 129 * (i + 1)]

    return w

def matricize_Tij(params):
    t_ij = params[26 * 129:]
    t_ij = t_ij.reshape((26, 26)).T
    return t_ij


def compute_gradient_wrt_Wy(X, y, w, t, alpha, beta, denominator):
    gradient = np.zeros((26, 129))

    w_x = np.dot(X,w.T)

    for i in range(len(X)):
        gradient[y[i]] += X[i]

        # for each position, reduce the probability of the character
        temp = np.ones((26, 129)) * X[i]
        temp = temp.transpose()
        temp = temp * np.exp((alpha[i] + beta[i] + w_x[i]) - denominator)

        gradient -= temp.transpose()

    return gradient.flatten()


def compute_gradient_wrt_Tij(y, x, w, t, alpha, beta, denominator):
    gradient = np.zeros(26 * 26)
    w_x = np.dot(x,w.T)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(w_x[i] + t.transpose()[j] + w_x[i + 1][j] + beta[i + 1][j] + alpha[i])

    # normalize the gradient
    gradient /= np.exp(denominator)

    # add the gradient for the next word
    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def gradient_per_word(X, y, w, t, word_index, concat_grads=True):
    # O(n * |Y|)
    # w_x = np.dot(X[word_index], w.T)
    # O(n * |Y|^2)
    f_mess = compute_forward_message(X[word_index], w, t)
    # O(n * |Y|^2)
    b_mess = compute_backward_message(X[word_index], w, t)
    # O(1)
    den = compute_denominator(f_mess, X[word_index], w)
    # O(n * |Y|^2)
    wy_grad = compute_gradient_wrt_Wy(X[word_index], y[word_index], w, t, f_mess, b_mess, den)
    # O(n * |Y|^2)
    t_grad = compute_gradient_wrt_Tij(y[word_index], X[word_index], w, t, f_mess, b_mess, den)

    if concat_grads:
        return np.concatenate((wy_grad, t_grad))
    else:
        wy_grad = wy_grad.reshape((26, 129))
        t_grad = t_grad.reshape((26, 26))
        return wy_grad, t_grad


def averaged_gradient(params, X, y, limit):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = np.zeros(129 * 26 + 26 ** 2)
    for i in range(limit):
        total += gradient_per_word(X, y, w, t, i)
    return total / (limit)


def compute_log_p_y_given_x(x,w, y, t, word_index):
    f_mess = compute_forward_message(x, w, t)
    return np.log(compute_numerator(y, x, w, t) / np.exp(compute_denominator(f_mess, x, w)))


def compute_log_p_y_given_x_avg(params, X, y, limit):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = 0
    for i in range(limit):
        # w_x = np.dot(X[i], w)
        total += compute_log_p_y_given_x(X[i],w, y[i], t, i)

    return total / (limit)


def check_gradient(params, X, y):
    # check the gradient of the first 10 words
    grad_value = check_grad(compute_log_p_y_given_x_avg, averaged_gradient, params, X, y, 1)
    print("Gradient check (first 10 character) : ", grad_value)


def measure_gradient_computation_time(params, X, y):
    # this takes 7.2 seconds for us
    start = time.time()
    av_grad = averaged_gradient(params, X, y, len(X))
    print("Total time:", time.time() - start)

    with open("result/gradient.txt", "w") as text_file:
        for i, elt in enumerate(av_grad):
            text_file.write(str(elt))
            text_file.write("\n")


def optimization_function(params, X, y, C):
    num_examples = len(X)
    l2_regularization = 1 / 2 * np.sum(params ** 2)
    log_loss = compute_log_p_y_given_x_avg(params, X, y, num_examples)
    return -C * log_loss + l2_regularization


def d_optimization_function(params, X, y, C):
    num_examples = len(X)
    logloss_gradient = averaged_gradient(params, X, y, num_examples)
    l2_loss_gradient = params
    return -C * logloss_gradient + l2_loss_gradient


def train_crf_lbfgs(params, X, y, C, model_name):
    print("Optimizing parameters. This will take a long time (at least 1 hour per model).")

    start = time.time()
    out = fmin_bfgs(optimization_function, params, d_optimization_function, (X, y, C), disp=1)
    print("Total time: ", end='')
    print(time.time() - start)

    with open("result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt) + "\n")


def train_crf_sgd(params, X, y, C, num_epochs, learning_rate, l2_lambda, test_xy, model_name):
    num_words = len(X)
    test_X, test_Y = test_xy

    W = matricize_W(params)
    T = matricize_Tij(params)

    print("Starting SGD optimizaion")

    # make results reproducible
    np.random.seed(1)

    start = time.time()
    for epoch in range(num_epochs):
        print("Begin epoch %d" % (epoch + 1))

        indices = np.arange(0, num_words, step=1, dtype=int)
        np.random.shuffle(indices)

        for i, word_index in enumerate(indices):
            W_grad, T_grad = gradient_per_word(X, y, W, T, word_index, concat_grads=False)

            # perform SGD update
            W -= learning_rate * W_grad + l2_lambda * W
            T -= learning_rate * T_grad + l2_lambda * T

            print("W norm", np.linalg.norm(W))
            print("T norm", np.linalg.norm(T))

        if (epoch + 1) % 10 == 0:
            print('*' * 80)
            print("Computing metrics after end of epoch %d" % (epoch + 1))
            # print evaluation metrics every 1000 steps of SGD
            train_loss = optimization_function(params, X, y, C)

            y_preds = decode_crf(test_X, W, T)
            word_acc, char_acc = compute_word_char_accuracy_score(y_preds, test_Y)

            print("Epoch %d | Train loss = %0.8f | Word Accuracy = %0.5f | Char Accuracy = %0.5f" %
                  (epoch + 1, train_loss, word_acc, char_acc))
            print('*' * 80, '\n')

    # merge the two grads into a single long vector
    out = np.concatenate((W.flatten(), T.flatten()))
    print("Total time: ", end='')
    print(time.time() - start)

    with open("result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt) + "\n")


def get_trained_model_parameters(model_name):
    file = open('result/' + model_name + '.txt', 'r')
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)


if __name__ == '__main__':

    ''' check gradients and write to file '''
    print("Checking gradient correctness")
    X, y = prepare_structured_dataset('train_sgd.txt')
    params = load_model_params()
    check_gradient(params, X, y)
    # measure_gradient_computation_time(params, X, y)

    #exit()

    ''' training  '''
    X_train, y_train = prepare_structured_dataset('train_sgd.txt')
    X_test, y_test = prepare_structured_dataset('test_sgd.txt')
    params = load_model_params_zeros()

    test_xy = [X_test, y_test]

    '''
    Run optimization. For C= 1000 it takes about an 56 minutes
    '''
    # Gradient based training (SGD) parameters
    NUM_EPOCHS = 1000
    LEARNING_RATE = 1e-2
    L2_LAMBDA = 1e-2

    train_crf_sgd(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
                  num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
                  test_xy=test_xy, model_name='sgd-2')

    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.005
    # L2_LAMBDA = 1e-4

    # train_crf_sgd(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='sgd-4')
    #
    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.005
    # L2_LAMBDA = 1e-6
    #
    # train_crf_sgd(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='sgd-6')

    params = get_trained_model_parameters('sgd-2')
    w = matricize_W(params)
    t = matricize_Tij(params)

    print("Function value: ", optimization_function(params, X_train, y_train, C=1))

    ''' accuracy '''
    y_preds = decode_crf(X_test, w, t)

    with open("result/prediction-sgd.txt", "w") as text_file:
        for i, elt in enumerate(y_preds):
            # convert to characters
            for word in elt:
                text_file.write(str(word + 1))
                text_file.write("\n")

    print("Test accuracy : ", compute_word_char_accuracy_score(y_preds, y_test))
