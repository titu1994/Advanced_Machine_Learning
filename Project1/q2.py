import csv
import numpy as np
import re
import collections
import sys
import timeit
import threading

from data_loader import load_Q2_data

Q2_MODEL_PATH = "data/model.txt"

def load_Q2_model():
    with open(Q2_MODEL_PATH, 'r') as f:
        lines = f.readlines()
    Wj = lines[:26*128]
    Tij = lines[26*128:]
    Wj = np.array(Wj, dtype=np.float32).reshape((26, 128))
    Tij = np.array(Tij, dtype=np.float32).reshape((26, 26), order='F')
    return Wj, Tij

# forward pass for j=0...m
def forward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    alpha = np.zeros([X_train.shape[0], Wj.shape[0]]) # init alpha values to zeros
    alpha[0,] = 1 # init first alpha[Xj,]  to 1
    
    for j in range(1, X_train.shape[0]):
        if X_train[j][0] != X_train[j-1][0]:
            alpha[j,] = 1 # init alpha[Xj,] to 1 if new word
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    alpha[j, s1] += alpha[j-1, s2] * np.exp(inter[s1] + Tij[s2, s1])
    stop = timeit.default_timer()
    print('forward_pass (s): ' + str(stop - start))

    # save computed alpha in alpha.npy
    np.save('alpha.npy', alpha)
    return alpha

# backward pass for j=m-1...0
def backward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    beta = np.zeros([X_train.shape[0], Wj.shape[0]])
    beta[-1,] = 1 # init last beta[Xj,] to 1
    
    for j in range(X_train.shape[0] - 2, -1, -1):
        if X_train[j][0] != X_train[j+1][0]:
            beta[j,] = 1 # init beta[Xj,] to 1 if last letter of new word
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    beta[j, s2] += beta[j+1, s1] * np.exp(inter[s2] + Tij[s1, s2])
                    # beta[j, s1] += beta[j+1, s2] * np.exp(inter[s1] + Tij[s2, s1])
    stop = timeit.default_timer()
    print('backward_pass (s): ' + str(stop - start))

    # save computed beta in beta.npy
    np.save('beta.npy', beta)
    return beta

def log_forward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    alpha = np.zeros([X_train.shape[0], Wj.shape[0]])
    alpha[0,] = 0
    for j in range(1, X_train.shape[0]):
        if X_train[j][0] != X_train[j-1][0]:
            alpha[j,] = 0
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    alpha[j, s1] = np.log(np.exp(alpha[j-1, s2] + inter[s1] + Tij[s2, s1]))
    stop = timeit.default_timer()
    print('log_forward_pass (s): ' + str(stop - start))
    np.save('log_alpha.npy', alpha)

    result = open(r'log_alpha.txt', 'w+')
    for j in range(X_train.shape[0]):
         result.write('la[' + str(j) + ']: ' + str([alpha[j, s] for s in range(1, Wj.shape[0])]) + '\n')
    result.close()

    return alpha

def log_backward_pass(X_train, Wj, Tij):
    start = timeit.default_timer()
    beta = np.zeros([X_train.shape[0], Wj.shape[0]])
    beta[-1,] = 0

    for j in range(X_train.shape[0] - 2, -1, -1):
        if X_train[j][0] != X_train[j+1][0]:
            beta[j,] = 0
        else:
            inter = np.dot(X_train[j][1:], Wj.T)
            for s1 in range(Wj.shape[0]):
                for s2 in range(Wj.shape[0]):
                    beta[j, s2] = np.log(np.exp(beta[j+1, s1] + inter[s2] + Tij[s2, s1]))
    stop = timeit.default_timer()
    print('log_backward_pass (s): ' + str(stop - start))
    np.save('log_beta.npy', beta)

    result = open(r'log_beta.txt', 'w+')
    for j in range(X_train.shape[0]):
         result.write('lb[' + str(j) + ']: ' + str([beta[j, s] for s in range(1, Wj.shape[0])]) + '\n')
    result.close()

    return beta

def gradient_Wj(dist, X_train, y_train, Wj):
    start = timeit.default_timer()
    gradient = np.zeros_like(Wj)  # (26, 128)

    for j in range(X_train.shape[0]):
        y_delta = (y_train[j] - dist[j]).reshape((-1, 1))
        x_delta = X_train[j, 1:].reshape((1, -1))
        gradient += np.dot(y_delta, x_delta)

    gradient /= len(X_train)

    flattened_gradient = gradient.flatten()

    result = open(r'grad.txt', 'w')
    for g in flattened_gradient:
        result.write(str(g) + "\n")
    result.close()

    stop = timeit.default_timer()
    print('gradient_Wj (s): ' + str(stop - start))

def gradient_Tij(dist_tj, X_train, y_train, Wj, Tij):
    start = timeit.default_timer()
    gradient = np.zeros_like(Tij)  # (26, 26)

    for j in range(X_train.shape[0] - 1):
        for s1 in range(Wj.shape[0]):
            for s2 in range(Wj.shape[0]):
                indicator = y_train[j][s1] and y_train[j+1][s2]
                gradient[s1, s2] += indicator - (dist_tj[j][s1] * dist_tj[j+1][s2])

    gradient /= len(X_train)
    flattened_gradient = gradient.flatten()

    result = open(r'grad_tij.txt', 'w')
    for g in flattened_gradient:
        result.write(str(g) + "\n")
    result.close()

    stop = timeit.default_timer()
    print('gradient_Tij (s): ' + str(stop - start))
                #     print(str(j) + ' and ' + str(j+1))
                # print('y_train[' + str(j) + '][' + str(s1) + ']: ' + str(y_train[j][s1]) + ', ' + 'y_train[' + str(j+1) + '][' + str(s2) + ']: ' + str(y_train[j+1][s2]))

    # result = open(r'grad.txt', 'a+')

    # flattened_gradient = gradient.flatten()
    # for g in flattened_gradient:
    #     result.write(str(g) + "\n")
    # result.close()

    stop = timeit.default_timer()
    print('gradient_Tij (s): ' + str(stop - start))

def conditional_prob_Tij(X_train, y_train, Wj, Tij):
    dist = np.zeros([X_train.shape[0], Wj.shape[0]])
    start = timeit.default_timer()

    alpha = np.load('alpha.npy', mmap_mode='r')
    beta = np.load('beta.npy', mmap_mode='r')

    for j in range(X_train.shape[0] - 1):
        inter = np.dot(X_train[j][1:], Wj.T)
        inter_next = np.dot(X_train[j+1][1:], Wj.T)
        for s1 in range(Wj.shape[0]):
            for s2 in range(Wj.shape[0]):
                alpha_temp = alpha[j, s1]
                beta_temp = beta[j+1, s2]
                dist[j, s1] += alpha_temp * beta_temp * np.exp(inter[s1] + inter_next[s2] + Tij[s1, s2])

    inter = np.exp(np.dot(X_train[-1][1:], Wj.T))
    for s in range(Wj.shape[0]):
        alpha_temp = alpha[-2, s1]
        beta_temp = 1
        dist[-1, s] = alpha_temp * beta_temp * np.exp(inter[s])

    distsum = dist.sum(axis=1)
    dist = dist / distsum[:, np.newaxis]
    avg = np.mean(np.log(dist))
    print("avg: " + str(avg))
    print(dist.shape)

    result = open(r'dist_tij.txt', 'w+')
    for j in range(X_train.shape[0]):
         result.write('dist[' + str(j) + ']: ' + str([dist[j, s] for s in range(Wj.shape[0])]) + '\n')
    result.close()

    gradient_Tij(dist, X_train, y_train, Wj, Tij)

    stop = timeit.default_timer()
    print('conditional_prob_Wj (s): ' + str(stop - start))

def conditional_prob_Wj(X_train, y_train, Wj, Tij):
    dist = np.zeros([X_train.shape[0], Wj.shape[0]])
    # dist[0,] = 1
    start = timeit.default_timer()

    # load alpha/beta from .npy files (pre-computed)
    alpha = np.load('alpha.npy', mmap_mode='r')
    beta = np.load('beta.npy', mmap_mode='r')
    # log_alpha = log_forward_pass(X_train, Wj, Tij)
    # log_beta = log_backward_pass(X_train, Wj, Tij)
    #alpha = forward_pass(X_train, Wj, Tij)
    #beta = backward_pass(X_train, Wj, Tij)

    for j in range(X_train.shape[0]):
        inter = np.exp(np.dot(X_train[j][1:], Wj.T))
        for s in range(Wj.shape[0]):
            alpha_temp = alpha[j, s]
            beta_temp = beta[j, s]
            dist[j, s] = alpha_temp * beta_temp * inter[s]

    # for j in range(X_train.shape[0]):
    #     inter = np.exp(np.dot(X_train[j][1:], Wj.T))
    #     for s1 in range(Wj.shape[0]):
    #         for s2 in range(Wj.shape[0]):
    #             alpha_temp = alpha[j, s2] # s2???
    #             beta_temp = beta[j, s1]
    #             dist[j, s1] = alpha_temp * beta_temp * inter[s1]
        # dist[j,] /= alpha[]

    # for j in range(X_train.shape[0] - 1):
    #     inter = np.exp(np.dot(X_train[j+1][1:], Wj.T))
    #     for s1 in range(Wj.shape[0]):
    #         for s2 in range(Wj.shape[0]):
    #             alpha_temp = alpha[j-1, s2]
    #             beta_temp = beta[j+1, s1]
    #             dist[j, s1] = alpha_temp * beta_temp * inter[s1]

    # inter = np.exp(np.dot(X_train[j][1:], Wj.T))
    # for s1 in range(Wj.shape[0]):
    #     for s2 in range(Wj.shape[0]):
    #         alpha_temp = alpha[-2, s2]
    #         beta_temp = 1
    #         dist[-1, s1] = alpha_temp * beta_temp * inter[s1]
    
    # Z = fm(ym) is last letter of every word...?
    #dist /= np.sum(alpha[-1])
    distsum = dist.sum(axis=1)
    dist = dist / distsum[:, np.newaxis]

    # 1/n*log p(y|X)
    avg = np.mean(np.log(dist))
    print("avg: " + str(avg))
    print(dist.shape)

    result = open(r'dist.txt', 'w+')
    for j in range(X_train.shape[0]):
         result.write('dist[' + str(j) + ']: ' + str([dist[j, s] for s in range(Wj.shape[0])]) + '\n')
    result.close()

    gradient_Wj(dist, X_train, y_train, Wj)

    stop = timeit.default_timer()
    print('conditional_prob_Wj (s): ' + str(stop - start))

if __name__ == '__main__':
    Wj, Tij = load_Q2_model()
    train_data, test_data = load_Q2_data()
    t_list = list(train_data.items())

    y_train = np.zeros([len(t_list), 26], dtype=np.int8) # [n, 26], change it to n
    X_train = np.zeros([len(t_list), 129], dtype=np.int16) # [n, 128], change it to n
    for index, i in enumerate(t_list):
        y_train[index, ord(i[1][0][0]) - 97] = 1
        X_train[index][0] = i[1][2]
        X_train[index][1:] = i[1][4]

    print("y_train shape: " + str(y_train.shape))
    print("X_train.shape: " + str(X_train.shape))
    print("Wj.shape: " + str(Wj.shape))
    print("Tij.shape: " + str(Tij.shape))

    conditional_prob_Wj(X_train, y_train, Wj, Tij)
    conditional_prob_Tij(X_train, y_train, Wj, Tij)

# store result in .txt
# result = open(r'dist.txt', 'w+')
# for j in range(0, X_train.shape[0]):
#     result.write('dist[' + str(j) + ']: ' + str([alpha[j, s] for s in range(1, Wj.shape[0])]) + '\n')
# result.close()

# # threading
# class my_thread(threading.Thread):
#     def __init__(self, threadID, name, X_train, Wj, Tij, p):
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.name = name
#         self.X_train = X_train
#         self.Wj = Wj
#         self.Tij = Tij
#         self.p = p
#     def run(self):
#         if self.p == 'fp':
#             forward_pass(X_train, Wj, Tij)
#         elif self.p == 'bp':
#             backward_pass(X_train, Wj, Tij)
#         else:
#             conditional_prob(X_train, Wj, Tij)
#fp = my_thread(1, 'fp-thread', X_train, Wj, Tij, 'fp')
#bp = my_thread(2, 'bp-thread', X_train, Wj, Tij, 'bp')
# cp = my_thread(3, 'cp-thread', X_train, Wj, Tij, 'cp')
# cp.start()
# fp.start()
# bp.start()