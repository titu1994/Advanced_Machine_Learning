import numpy as np

from Project1.data_loader import load_Q1_data

X, W, Tij = load_Q1_data()

num_labels = 26
m = X.shape[0]  # 100
dim = X.shape[1]  # 128

def brute_force():
    # compute <W_yj, Xj>
    wx_sum = np.zeros((m, 26,), dtype=np.float32)
    for j in range(m):
        for y_j in range(num_labels):
            wx_sum[j, y_j] += np.dot(X[j], W[y_j].T)

    # compute T[y_j, y_j + 1]
    t_sum = np.zeros((1, 26), dtype=np.float32)
    for y_j in range(num_labels):
        if y_j + 1 != num_labels:
            t_sum[0, y_j] += Tij[y_j, y_j + 1]

    t_sum = np.repeat(t_sum, m, axis=0)
    t_sum[-1, :] = 0.0

    y_hat = wx_sum + t_sum

    predictions = np.argmax(y_hat, axis=-1)
    objective_value = np.max(y_hat, axis=-1)

    return predictions, objective_value


def brute_force_vectorized():
    # compute <W_yj, Xj>
    wx_sum = np.dot(X, W.T)

    # compute T[y_j, y_j + 1]
    t_sum = np.zeros((1, 26), dtype=np.float32)
    for y_j in range(num_labels):
        t_sum[0] += Tij[y_j]

    t_sum = np.repeat(t_sum, m, axis=0)
    t_sum[-1, :] = 0.0

    y_hat = wx_sum + t_sum

    predictions = np.argmax(y_hat, axis=-1)
    objective_value = np.max(y_hat, axis=-1)

    return predictions, objective_value


def max_sum_algorithm():
    pass


if __name__ == '__main__':
    preds, objective = brute_force_vectorized()

    for i, (p, o) in enumerate(zip(preds, objective)):
        print("i=%d : Predicted = %s (objective = %0.2f)" % (i + 1, chr(p + ord('a')), o))

    print("Maximum objective score : ", objective.max())