import numpy as np
from scipy.optimize import check_grad, fmin_bfgs
import time

from utils import read_data_formatted, get_params, flatten_dataset, reshape_dataset, compute_accuracy
from q1 import decode_crf

def forward_propogate(w_x, t):
    word_len = len(w_x)
    # establish matrix to hold results
    M = np.zeros((word_len, 26))
    # set first row to inner <wa, x0> <wb, x0>...

    # iterate through length of word
    for i in range(1, word_len):
        vect = M[i - 1] + t.transpose()
        vect_max = np.max(vect, axis=1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + w_x[i - 1]), axis=1))

    return M


def back_propogate(w_x, t):
    # get the index of the final letter of the word
    fin_index = len(w_x) - 1

    # only need to go from the end to stated position
    M = np.zeros((len(w_x), 26))
    # now we need taa, tab, tac... because we are starting at the end and working backwards
    # which is exactly the transposition of the t matrix
    t_trans = t

    for i in range(fin_index - 1, -1, -1):
        vect = M[i + 1] + t_trans
        vect_max = np.max(vect, axis=1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + w_x[i + 1]), axis=1))
    return M


def num_letter(w_x, f_mess, b_mess, position, letter):
    factor = 0
    # if(position > 0):
    factor += f_mess[position][letter]

    # if(position < len(w_x) -1):
    factor += b_mess[position][letter]

    return np.exp(factor + w_x[position][letter])


def numerator(y, w_x, t):
    # full numerator for an entire word
    total = 0
    # go through whole word
    for i in range(len(w_x)):
        # no matter what add W[actual letter] inner Xi
        total += w_x[i][y[i]]
        if (i > 0):
            # again we have t stored as Tcur, prev
            total += t[y[i - 1]][y[i]]
    return np.exp(total)


def denominator(f_message, w_x):
    # this is  eassy, just forward propogate to the end of the word and return the sum of the exponentials
    return np.sum(np.exp(f_message[-1] + w_x[-1]))


# split up params into w and t.  Note that this only needs to happen once per word!!! do not calculate per letter
def w_matrix(params):
    w = np.zeros((26, 128))
    for i in range(26):
        w[i] = params[128 * i: 128 * (i + 1)]
    return w


def t_matrix(params):
    t = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            # want to be able to say t[0] and get all values of Taa, tba, tca...
            t[j][i] = params[128 * 26 + count]
            count += 1
    return t


def grad_wrt_wy(X, y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros((26, 128))
    for i in range(len(X)):
        gradient[y[i]] += X[i]
        # for each position subtract off the probability of the letter
        temp = np.ones((26, 128)) * X[i]
        temp = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
        gradient -= temp.transpose()
    return gradient.flatten()


def grad_wrt_t(y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(26 * 26)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(
                w_x[i] + t.transpose()[j] + w_x[i + 1][j] + b_mess[i + 1][j] + f_mess[i])

    gradient /= den

    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def gradient_word(X, y, w, t, word_num):
    # print()
    # O(n|Y|)
    # start = time.time()
    w_x = np.inner(X[word_num], w)
    # print("w_x time : " + str(time.time() - start))
    # O(n{Y}^2)

    # start = time.time()
    f_mess = forward_propogate(w_x, t)
    # print("f_prop_time : " + str(time.time() - start))

    # O(n{Y}^2)
    # start = time.time()
    b_mess = back_propogate(w_x, t)
    # print("b_prop_time : " + str(time.time() - start))

    # O(1)
    # start = time.time()
    den = denominator(f_mess, w_x)
    # print("den_time : " + str(time.time() - start))

    # O(n|Y|^2)
    # start = time.time()
    wy_grad = grad_wrt_wy(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den)
    # print("grad_wy time : " + str(time.time() - start))

    # O(n|Y|^2)
    # start = time.time()
    t_grad = grad_wrt_t(y[word_num], w_x, t, f_mess, b_mess, den)
    # print("grad_t time : " + str(time.time() - start))

    return np.concatenate((wy_grad, t_grad))


def gradient_avg(params, X, y, up_to_index):
    w = w_matrix(params)
    t = t_matrix(params)

    total = np.zeros(128 * 26 + 26 ** 2)
    for i in range(up_to_index):
        total += gradient_word(X, y, w, t, i)
    return total / (up_to_index)


def log_p_y_given_x(w_x, y, t, word_num):
    f_mess = forward_propogate(w_x, t)
    return np.log(numerator(y, w_x, t) / denominator(f_mess, w_x))


def log_p_y_given_x_avg(params, X, y, up_to_index):
    w = w_matrix(params)
    t = t_matrix(params)

    total = 0
    for i in range(up_to_index):
        w_x = np.inner(X[i], w)
        total += log_p_y_given_x(w_x, y[i], t, i)
    return total / (up_to_index)


def check_gradient(params, X, y):
    # kind of like a unit test.  Just check the gradient of the first 10 words
    # this takes a while so be forwarned
    print("Gradient check (first 10 characters) : ",
          check_grad(log_p_y_given_x_avg,
                     gradient_avg,
                     params, X, y, 10))


def timed_gradient_calculation(params, X, y):
    # this takes 7.2 seconds for me
    start = time.time()
    av_grad = gradient_avg(params, X, y, len(X))
    print("Total time:", time.time() - start)

    with open("result/gradient.txt", "w") as text_file:
        for i, elt in enumerate(av_grad):
            text_file.write(str(elt))
            text_file.write("\n")


def optimization_function(params, X, y, C):
    num_examples = len(X)
    reg = 1 / 2 * np.sum(params ** 2)
    avg_prob = log_p_y_given_x_avg(params, X, y, num_examples)
    return -C * avg_prob + reg


def grad_func(params, X, y, C):
    num_examples = len(X)
    grad_avg = gradient_avg(params, X, y, num_examples)
    grad_reg = params
    return -C * grad_avg + grad_reg


def optimize(params, X, y, C, name):
    print("Optimizing parameters. This will take a long time (at least 1 hours).")
    start = time.time()
    out = fmin_bfgs(optimization_function, params, grad_func, (X, y, C), disp=1)
    print("Total time: ", end='')
    print(time.time() - start)

    with open("result/" + name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt) + "\n")


def get_optimal_params(name):
    file = open('result/' + name + '.txt', 'r')
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)


if __name__ == '__main__':

    ''' check gradients and write to file '''
    #X, y = read_data_formatted('train_struct.txt')
    #params = get_params()
    #check_gradient(params, X, y)
    #timed_gradient_calculation(params, X, y)

    '''  '''

    ''' optimization '''
    X_train, y_train = read_data_formatted('train_struct.txt')
    X_test, y_test = read_data_formatted('test_struct.txt')
    params = get_params()

    ''' 
    Run optimization.  For C= 1000 it takes about an 56 minutes
    Commented out, as the next step will read the prepared optimized weights
    '''
    # optimize(params, X_train, y_train, C=1000, name='solution1000')

    params = get_optimal_params('solution')
    w = w_matrix(params)
    t = t_matrix(params)

    print("Function value: ", optimization_function(params, X_train, y_train, C=1000))

    x_test = flatten_dataset(X_test)

    ''' accuracy '''
    y_preds = decode_crf(x_test, w, t)

    y_preds = reshape_dataset(y_preds, y_test)

    with open("result/prediction.txt", "w") as text_file:
        for i, elt in enumerate(y_preds):
            # convert to characters
            for word in elt:
                text_file.write(str(word))
                text_file.write("\n")

    print("Test accuracy : ", compute_accuracy(y_preds, y_test))









