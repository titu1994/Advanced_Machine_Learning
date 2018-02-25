import numpy as np

from Project1.data_loader import load_Q1_data

X, W, Tij = load_Q1_data()

num_labels = 26
m = X.shape[0]  # 100
dim = X.shape[1]  # 128

def brute_force():
    y_hat = np.zeros((m, num_labels), dtype=np.float32)

    # compute <W_yj, Xj>
    wx_sum = np.zeros((m, 26,), dtype=np.float32)
    for j in range(m):
        for y_j in range(num_labels):
            wx_sum[j, y_j] += np.dot(X[j], W[y_j].T)


    t_sum = np.zeros((m, 26,), dtype=np.float32)
    for j in range(m):
        for y_j in range(num_labels - 1):
                t_sum[j, y_j] += Tij[y_j, y_j + 1]

    y_hat += wx_sum + t_sum

    predictions = np.argmax(y_hat, axis=-1)
    confidence = np.max(y_hat, axis=-1)

    return predictions, confidence


def max_sum_algorithm():
    pass


if __name__ == '__main__':
    preds, confidence = brute_force()

    for i, (p, c) in enumerate(zip(preds, confidence)):
        print("i=%d : Predicted = %s (confidence = %0.2f)" % (i + 1, chr(p + ord('a')), c))