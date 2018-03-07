import csv
import numpy as np
import re
import timeit
import pdb
from data_loader import load_Q2_model, load_Q2_data


def get_pixels(Xword):
    Xword_hold = np.zeros([len(Xword), 128])  # init zeros to hold current word pixel values
    for i in range(len(Xword)):
        Xword_hold[i] = Xword[i][2]  # XWord_hold has pixel values for only letters in current word
    return Xword_hold


def forward_pass(Xword, Wj, Tij):  # calculate alpha(j, s)
    Xword_alpha = np.zeros([len(Xword), 26])  # init (m x 26) matrix to hold alpha values
    Xword = get_pixels(Xword)  # get pixel values for current letters in word
    Xword = np.inner(Xword, Wj)  # inner product of XWord and Wj weights

    for j in range(1, len(Xword)):  # for each letter j in XW m
        letter_alpha = Xword_alpha[j - 1] + Tij.T
        letter_alpha_max = np.max(letter_alpha, axis=1)
        letter_alpha = (letter_alpha.T - letter_alpha_max).T
        Xword_alpha[j] = letter_alpha_max + np.log(np.sum(np.exp(letter_alpha + Xword[j - 1]), axis=1))

    return Xword_alpha  # return alpha values for j=0..m for XW


def backward_pass(Xword, Wj, Tij):  # calculate beta(j, s)
    Xword_beta = np.zeros([len(Xword), 26])
    Tij = Tij.T
    Xword = get_pixels(Xword)  # get pixel values for current letters in word
    Xword = np.inner(Xword, Wj)  # inner product of XWord and Wj weights

    for j in range(len(Xword) - 2, -1, -1):
        letter_beta = Xword_beta[j + 1] + Tij.T
        letter_beta_max = np.max(letter_beta, axis=1)
        letter_beta = (letter_beta.T - letter_beta_max).T
        Xword_beta[j] = letter_beta_max + np.log(np.sum(np.exp(letter_beta + Xword[j + 1]), axis=1))

    return Xword_beta  # return beta values for j=m-1..0 for XW


def psi_fn(Xword, y, Tij):  # psi_fn (numerator) for calculating likelihood value
    final = 0
    for j in range(len(Xword)):
        if j > 0:
            final += Tij[y[j]][y[j - 1]]
    return np.exp(final)


def partition_fn(Xword, alpha):  # Z (denominator) that normalizes psi_fn
    return np.sum(np.exp(alpha[-1] + Xword[-1]))


def logpy_X(Xword, y, Tij):  # logp(y|X) for each word
    alpha = forward_pass(Xword, Tij)
    return np.log(psi_fn(Xword, y, Tij) / partition_fn(Xword, alpha))


def logpy_X_sum(Wj, Tij, X_train, y, index):  # logsum(p|X) for all words
    final = 0
    for j in range(index):
        Xword = np.inner(X_train[j], Wj)
        final += logpy_X(Xword, y[j], Tij)
    return final


def gradWj(X, y, Xword, Tij, alpha, beta, partition):
    grad = np.zeros((26, 128))

    for j in range(len(X)):
        grad[y[j]] += X[j]
        temp = np.ones((26, 128)) * X[i]
        temp = temp.T * np.exp(alpha[j] + beta[j] + Xword[j]) / partition
        grad -= temp.T

    return grad.flatten()


def gradTij(Xword, y, Tij, alpha, beta, partition):
    grad = np.zeros(26 * 26)
    for j in range(len(Xword) - 1):
        for s in range(26):
            grad[s * 26:(s + 1) * 26] -= np.exp(Xword[j] + Tij.T[s] + Xword[j + 1][s] + beta[j + 1][s] + alpha[j])
    grad /= partition

    for j in range(len(Xword) - 1):
        Tij_index = y[j]
        Tij_index += 26 * y[j + 1]
        grad[Tij_index] += 1

    return grad


def gradWord(X_train, y, Wj, Tij, j):
    Xword = np.inner(X_train[j], Wj)
    alpha = forward_pass(Xword, Tij)
    beta = backward_pass(Xword, Tij)
    partition = partition_fn(alpha, Xword)
    Wj_grad = gradWj(X_train[j], y[j], Xword, Tij, alpha, beta, partition)
    Tij_grad = gradTij(y[j], Xword, Tij, alpha, beta, partition)
    return np.concatenate((Wj_grad, Tij_grad))


def gradAvg(X_train, y, Wj, Tij, index):
    final = np.zeros(128 * 26 + 26 ** 2)
    for j in range(index):
        final += gradWord(X_train, y, Wj, Tij, j)
    return final / (index)


if __name__ == '__main__':
    Wj, Tij = load_Q2_model()
    X_train = load_Q2_data()
