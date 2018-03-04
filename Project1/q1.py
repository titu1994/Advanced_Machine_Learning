import numpy as np

from data_loader import load_Q1_data

X, W, Tij = load_Q1_data()

num_labels = 26
m = X.shape[0]  # 100
dim = X.shape[1]  # 128

def brute_force_vectorized():
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


def max_sum_decoder():
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
    preds, objective = brute_force_vectorized()

    # for i, (p, o) in enumerate(zip(preds, objective)):
    #     print("i=%d : Predicted = %s (objective = %0.2f)" % (i + 1, chr(p + ord('a')), o))

    preds_max_sum = max_sum_decoder()

    for i,p in enumerate(preds_max_sum):
        print("i=%d : Predicted = %s" % (i+1, chr(p+ord('a'))))

    # print("Maximum objective score : ", objective.max())
    # print(max_sum_decoder())
