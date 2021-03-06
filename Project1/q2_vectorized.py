import csv
import numpy as np
import re
import collections
import sys
import timeit
import pdb

from scipy.optimize import fmin_l_bfgs_b, check_grad

from data_loader import load_Q2_data
from utils import to_categorical
from q1 import max_sum_decoder

Q2_MODEL_PATH = "data/model.txt"


def load_Q2_model():
    with open(Q2_MODEL_PATH, 'r') as f:
        lines = f.readlines()
    Wj = lines[:26 * 128]
    Tij = lines[26 * 128:]
    Wj = np.array(Wj, dtype=np.float32).reshape((26, 128))
    Tij = np.array(Tij, dtype=np.float32).reshape((26, 26), order='F')
    return Wj, Tij


def forward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    alpha = np.zeros([X_train.shape[0], Wj.shape[0]])  # init alpha values to zeros
    alpha[0,] = 1  # init first alpha[Xj,]  to 1

    for j in range(1, X_train.shape[0]):
        if X_train[j][0] != X_train[j - 1][0]:
            alpha[j,] = 1  # init alpha[Xj,] to 1 if new word
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    alpha[j, s1] += alpha[j - 1, s2] * np.exp(inter[s1] + Tij[s2, s1])
    stop = timeit.default_timer()
    #print('forward_pass (s): ' + str(stop - start))

    # save computed alpha in alpha.npy
    np.save('alpha.npy', alpha)
    return alpha


# backward pass for j=m-1...0
def backward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    beta = np.zeros([X_train.shape[0], Wj.shape[0]])
    beta[-1,] = 1  # init last beta[Xj,] to 1

    for j in range(X_train.shape[0] - 2, -1, -1):
        if X_train[j][0] != X_train[j + 1][0]:
            beta[j,] = 1  # init beta[Xj,] to 1 if last letter of new word
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    beta[j, s2] += beta[j + 1, s1] * np.exp(inter[s2] + Tij[s1, s2])
                    # beta[j, s1] += beta[j+1, s2] * np.exp(inter[s1] + Tij[s2, s1])
    stop = timeit.default_timer()
    #print('backward_pass (s): ' + str(stop - start))

    # save computed beta in beta.npy
    np.save('beta.npy', beta)
    return beta


def log_backward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    beta = np.zeros([X_train.shape[0], Wj.shape[0]])
    beta[-1,] = 0

    for j in range(X_train.shape[0] - 2, -1, -1):
        if X_train[j][0] != X_train[j + 1][0]:
            beta[j,] = 0
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    beta[j, s2] = np.log(np.exp(beta[j + 1, s1] + inter[s2] + Tij[s2, s1]))
    stop = timeit.default_timer()
    print('log_backward_pass (s): ' + str(stop - start))
    np.save('log_beta.npy', beta)

    result = open(r'log_beta.txt', 'w+')
    for j in range(X_train.shape[0]):
        result.write('lb[' + str(j) + ']: ' + str([beta[j, s] for s in range(1, Wj.shape[0])]) + '\n')
    result.close()

    return beta


def gradient_Wj(dist, X, y, Wj):
    start = timeit.default_timer()
    # gradient = np.zeros_like(Wj)  # (26, 128)

    if y.ndim < 2:
        y = to_categorical(y, 26)

    x_index = np.arange(0, X.shape[0])
    y_deltas = (y - dist[x_index]).reshape((len(x_index), -1)).T
    x_deltas = X[x_index, 1:].reshape((len(x_index), -1))

    gradient = np.dot(y_deltas, x_deltas)
    #print(gradient.shape)

    # for j in range(X_train.shape[0]):
    #     y_delta = (y_train[j] - dist[j]).reshape((-1, 1))
    #     x_delta = X_train[j, 1:].reshape((1, -1))
    #     gradient += np.dot(y_delta, x_delta)

    gradient /= len(X)

    # for l2 regularization
    gradient += Wj
    #flattened_gradient = gradient.flatten()

    # result = open(r'grad.txt', 'w')
    # for g in flattened_gradient:
    #     result.write(str(g) + "\n")
    # result.close()

    stop = timeit.default_timer()
    #print('gradient_Wj (s): ' + str(stop - start))

    return gradient


def gradient_Tij(dist_tj, X, y, Wj, Tij):
    start = timeit.default_timer()
    gradient = np.zeros_like(Tij)  # (26, 26)

    lower_index = np.arange(0, X.shape[0] - 1)
    higher_index = np.arange(1, X.shape[0])

    # for j in range(X_train.shape[0] - 1):
    for s1 in range(Wj.shape[0]):
        for s2 in range(Wj.shape[0]):
            indicator = np.logical_and(y_train[lower_index, s1], y_train[higher_index, s2])
            gradient[s1, s2] = np.sum(indicator - (dist_tj[lower_index, s1] * dist_tj[higher_index, s2]))

    gradient /= len(X)
    flattened_gradient = gradient.flatten()

    # result = open(r'grad_tij.txt', 'w')
    # for g in flattened_gradient:
    #     result.write(str(g) + "\n")
    # result.close()

    stop = timeit.default_timer()
    #print('gradient_Tij (s): ' + str(stop - start))
    #     print(str(j) + ' and ' + str(j+1))
    # print('y_train[' + str(j) + '][' + str(s1) + ']: ' + str(y_train[j][s1]) + ', ' + 'y_train[' + str(j+1) + '][' + str(s2) + ']: ' + str(y_train[j+1][s2]))

    # result = open(r'grad.txt', 'a+')

    # flattened_gradient = gradient.flatten()
    # for g in flattened_gradient:
    #     result.write(str(g) + "\n")
    # result.close()

    stop = timeit.default_timer()
    #print('gradient_Tij (s): ' + str(stop - start))

    return gradient


def conditional_prob_Tij(X_train, y_train, Wj, Tij):
    dist = np.zeros([X_train.shape[0], Wj.shape[0]])
    start = timeit.default_timer()

    #alpha = np.load('alpha.npy', mmap_mode='r')
    #beta = np.load('beta.npy', mmap_mode='r')

    alpha = forward_pass(X_train, Wj, Tij)
    beta = backward_pass(X_train, Wj, Tij)

    lower_index = np.arange(0, X_train.shape[0] - 1)
    higher_index = np.arange(1, X_train.shape[0])

    inter_energies = np.dot(X_train[:, 1:], Wj.T)

    inter_lower = inter_energies[lower_index].astype('float64')
    inter_higher = inter_energies[higher_index].astype('float64')

    alpha_temp = alpha[lower_index].astype('float64')
    beta_temp = beta[higher_index].astype('float64')

    # for j in range(X_train.shape[0] - 1):
    for s1 in range(Wj.shape[0]):
        for s2 in range(Wj.shape[0]):
            # alpha_temp = alpha[lower_index]
            # beta_temp = beta[higher_index]
            dist[:-1, s1] += alpha_temp[:, s1] * beta_temp[:, s2] * np.exp(inter_lower[:, s1] + inter_higher[:, s2] + Tij[s1, s2])
            # for s2 in range(Wj.shape[0]):

            #   dist[j, s1] += alpha_temp * beta_temp * np.exp(inter[s1] + inter_next[s2] + Tij[s1, s2])

    inter = inter_energies[-1]  # np.exp(np.dot(X_train[-1][1:], Wj.T))
    for s in range(Wj.shape[0]):
        alpha_temp = alpha[-2, s]
        beta_temp = 1
        dist[-1, s] = alpha_temp * beta_temp * np.exp(inter[s])

    #distsum = dist.sum(axis=1)
    # dist = dist / distsum[:, np.newaxis]
    dist /= alpha

    avg = np.mean(np.log(dist))

    #print("avg: " + str(avg))
    print(dist.shape)

    # result = open(r'dist_tij.txt', 'w+')
    # for j in range(X_train.shape[0]):
    #      result.write('dist[' + str(j) + ']: ' + str([dist[j, s] for s in range(Wj.shape[0])]) + '\n')
    # result.close()

    gradient = gradient_Tij(dist, X_train, y_train, Wj, Tij)

    stop = timeit.default_timer()
    #print('conditional_prob_Wj (s): ' + str(stop - start))

    return dist, gradient


def conditional_prob_Wj(X_train, y_train, Wj):
    dist = np.zeros([X_train.shape[0], Wj.shape[0]])
    # dist[0,] = 1
    start = timeit.default_timer()

    # load alpha/beta from .npy files (pre-computed)
    #alpha = np.load('alpha.npy', mmap_mode='r')
    #beta = np.load('beta.npy', mmap_mode='r')
    alpha = forward_pass(X_train, Wj, Tij)
    beta = backward_pass(X_train, Wj, Tij)

    inter_energies = np.exp(np.dot(X_train[:, 1:], Wj.T))

    for s in range(Wj.shape[0]):
        alpha_temp = alpha[:, s]
        beta_temp = beta[:, s]
        dist[:, s] = alpha_temp * beta_temp * inter_energies[:, s]

    word_counter = 0
    for j in range(1, X_train.shape[0]):
        if X_train[j][0] != X_train[j-1][0]:
            for c in range(word_counter, -1, -1):
                dist[j-c] /= alpha[j]
            word_counter = 0
        else:
            word_counter += 1

    result = open(r'dist-new.txt', 'w+')
    for j in range(X_train.shape[0]):
        result.write('dist[' + str(j) + ']: ' + str([dist[j, s] for s in range(1, Wj.shape[0])]) + '\n')
    result.close()

    # Z = fm(ym) is last letter of every word...?
    # dist /= alpha
    #distsum = dist.sum(axis=1)
    #dist = dist / distsum[:, np.newaxis]
    avg = np.mean(np.log(dist))
    #print("avg: " + str(avg))
    print(dist.shape)
    gradient = gradient_Wj(dist, X_train, y_train, Wj)
    stop = timeit.default_timer()
    #print('conditional_prob_Wj (s): ' + str(stop - start))

    return dist, gradient

def log_conditional_prob(X_train, y_train, Wj, Tij):
    alpha = forward_pass(X_train, Wj, Tij)

def objective(w_t, x, y):
    w = w_t[:26*128].reshape((26, 128))
    t = w_t[26*128:].reshape((26, 26))

    preds, probas = max_sum_decoder(x[:, 1:], w, t)
    # probas = conditional_prob_Wj(x, y, w)
    # probas = np.log(probas)

    logloss = 1000 / len(x) * np.sum(probas)
    l2_loss = 1. * np.linalg.norm(w)
    transition_loss = 1. * np.sum(np.square(t))

    total_loss = logloss

    #print("Objective loss : ", total_loss)

    return total_loss

def d_objective(w_t, x, y):
    w = w_t[:26 * 128].reshape((26, 128))
    t = w_t[26 * 128:].reshape((26, 26))

    x = x.reshape((-1, 129))

    y_dash, y_probas = max_sum_decoder(x[:, 1:], w, t)
    #print('tshape', y_probas.shape)
    _, gradient_W = conditional_prob_Wj(x, y_probas, w)
    _, gradient_T = conditional_prob_Tij(x, y_probas, w, t)

    gradient_W = gradient_W.flatten()
    gradient_T = gradient_T.flatten()
    gradient = np.concatenate([gradient_W, gradient_T])
    return gradient

def max_sum_decoder_vector(w_t, X, Y):

    W = w_t[:26 * 128].reshape((26, 128))
    Tij = w_t[26 * 128:].reshape((26, 26))

    X = X[:, 1:]

    opt = np.zeros([X.shape[0], W.shape[0]], dtype=np.int32)
    nodeweights = np.zeros([X.shape[0], W.shape[0]])
    values = np.zeros([X.shape[0], W.shape[0]])
    for j in range(W.shape[0]):
        nodeweights[0, j] = np.dot(X[0], W[j])
    values[0] = nodeweights[0].copy()
    for i in range(1, X.shape[0]):
        nodeweights[i] = np.dot(X[i], W.T)
        for j in range(W.shape[0]):
            for k in range(W.shape[0]):
                val = values[i - 1, k] + nodeweights[i, j] + Tij[k, j]
                if values[i, j] < val:
                    values[i, j] = val
                    opt[i, j] = k
    k = np.argmax(nodeweights[X.shape[0] - 1])
    predictions = np.zeros([X.shape[0]], dtype=np.int32)
    predictions[X.shape[0] - 1] = k
    for i in range(X.shape[0] - 1, 0, -1):
        k = opt[i, k]
        predictions[i - 1] = k

    return predictions

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    Wj, Tij = load_Q2_model()
    train_data, test_data = load_Q2_data()
    t_list = list(train_data.items())

    y_train = np.zeros([len(t_list), 26], dtype=np.int8)  # [n, 26], change it to n
    X_train = np.zeros([len(t_list), 129], dtype=np.int16)  # [n, 128], change it to n
    for index, i in enumerate(t_list):
        y_train[index, ord(i[1][0][0]) - 97] = 1
        X_train[index][0] = i[1][2]
        X_train[index][1:] = i[1][4]

    print("y_train shape: " + str(y_train.shape))
    print("X_train.shape: " + str(X_train.shape))
    print("Wj.shape: " + str(Wj.shape))
    print("Tij.shape: " + str(Tij.shape))

    # conditional_prob_Wj(X_train, y_train, Wj)
    # conditional_prob_Tij(X_train, y_train, Wj, Tij)

    W_initial = np.zeros_like(Wj).flatten()
    T_initial = np.zeros_like(Tij).flatten()
    initial_guess = np.concatenate([W_initial, T_initial])

    EVAL_COUNT = 5
    #
    # result = fmin_l_bfgs_b(objective, initial_guess, d_objective, args=(X_train, y_train),
    #                        disp=1, maxfun=EVAL_COUNT, maxiter=EVAL_COUNT)

    # X_train = X_train[:2, :]
    # y_train = y_train[:2, :]
    # print(result)
    result = check_grad(objective, d_objective, initial_guess, X_train, y_train)
    # conditional_prob_Wj(X_train, y_train, Wj)
    print(result)