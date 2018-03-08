import numpy as np
import time
from scipy.optimize import check_grad, fmin_bfgs

from utils import prepare_structured_dataset, load_model_params, convert_word_to_character_dataset, convert_character_to_word_dataset, compute_word_char_accuracy_score
from q1 import decode_crf

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


def compute_denominator(alpha, x, w):
    # forward propagate to the end of the word and return the sum
    return np.sum(np.exp(alpha[-1] + np.dot(x,w.T)[-1]))


# convert W into a matrix from its flattened parameters
def matricize_W(params):
    w = np.zeros((26, 128))

    for i in range(26):
        w[i] = params[128 * i: 128 * (i + 1)]

    return w


def matricize_Tij(params):
    t_ij = np.zeros((26, 26))

    index = 0
    for i in range(26):
        for j in range(26):
            t_ij[j][i] = params[128 * 26 + index]
            index += 1

    return t_ij


def compute_gradient_wrt_Wy(X, y, w, t, alpha, beta, denominator):
    gradient = np.zeros((26, 128))

    w_x = np.dot(X,w.T)

    for i in range(len(X)):
        gradient[y[i]] += X[i]

        # for each position, reduce the probability of the character
        temp = np.ones((26, 128)) * X[i]
        temp = temp.transpose()
        temp = temp * np.exp(alpha[i] + beta[i] + w_x[i]) / denominator

        gradient -= temp.transpose()

    return gradient.flatten()


def compute_gradient_wrt_Tij(y, x, w, t, alpha, beta, denominator):
    gradient = np.zeros(26 * 26)
    w_x = np.dot(x,w.T)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(w_x[i] + t.transpose()[j] + w_x[i + 1][j] + beta[i + 1][j] + alpha[i])

    # normalize the gradient
    gradient /= denominator

    # add the gradient for the next word
    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def gradient_word(X, y, w, t, word_index):
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
    return np.concatenate((wy_grad, t_grad))


def averaged_gradient(params, X, y, limit):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = np.zeros(128 * 26 + 26 ** 2)
    for i in range(limit):
        total += gradient_word(X, y, w, t, i)
    return total / (limit)


def compute_log_p_y_given_x(x,w, y, t, word_index):
    f_mess = compute_forward_message(x, w, t)
    return np.log(compute_numerator(y, x, w, t) / compute_denominator(f_mess, x, w))


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
    grad_value = check_grad(compute_log_p_y_given_x_avg, averaged_gradient, params, X, y, 10)
    print("Gradient check (first 10 characters) : ", grad_value)


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


def train_crf(params, X, y, C, model_name):
    print("Optimizing parameters. This will take a long time (at least 1 hour per model).")

    start = time.time()
    out = fmin_bfgs(optimization_function, params, d_optimization_function, (X, y, C), disp=1)
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
    X, y = prepare_structured_dataset('train_struct.txt')
    params = load_model_params()
    check_gradient(params, X, y)
    measure_gradient_computation_time(params, X, y)

    ''' training  '''
    X_train, y_train = prepare_structured_dataset('train_struct.txt')
    X_test, y_test = prepare_structured_dataset('test_struct.txt')
    params = load_model_params()

    ''' 
    Run optimization. For C= 1000 it takes about an 56 minutes
    '''
    # optimize(params, X_train, y_train, C=1000, name='solution')

    params = get_trained_model_parameters('solution')
    w = matricize_W(params)
    t = matricize_Tij(params)

    print("Function value: ", optimization_function(params, X_train, y_train, C=1000))

    ''' accuracy '''
    x_test = convert_word_to_character_dataset(X_test)
    y_preds = decode_crf(x_test, w, t)
    y_preds = convert_character_to_word_dataset(y_preds, y_test)

    with open("result/prediction.txt", "w") as text_file:
        for i, elt in enumerate(y_preds):
            # convert to characters
            for word in elt:
                text_file.write(str(word + 1))
                text_file.write("\n")

    print("Test accuracy : ", compute_word_char_accuracy_score(y_preds, y_test))
