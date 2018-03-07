import numpy as np

from data_loader import load_Q1_data

X, W, Tij = load_Q1_data()

num_labels = 26
m = X.shape[0]  # 100
dim = X.shape[1]  # 128

def max_sum_decoder():
    opt = np.zeros([X.shape[0], W.shape[0]], dtype=np.int32)
    nodeweights = np.zeros([X.shape[0], W.shape[0]])
    values = np.zeros([X.shape[0], W.shape[0]])
    objective = np.zeros((X.shape[0]))
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
    objective[X.shape[0]-1] = values[X.shape[0]-1, k]
    for i in range(X.shape[0] - 1, 0, -1):
        k = opt[i, k]
        predictions[i - 1] = k
        objective[i-1] = values[i,k]
    return predictions,objective


def brute_force(X,W,T):
    predictions = np.zeros(X.shape[0], dtype=np.int32)
    tempsolutions = np.zeros(X.shape[0], dtype=np.int32)
    value = compute_brute_value(tempsolutions, X, W, T)
    while True:
        i=0
        while True:
            tempsolutions[i] += 1
            if(tempsolutions[i] == W.shape[0]):
                tempsolutions[i] = 0
                i+=1
            else:
                break
        newvalue = compute_brute_value(tempsolutions, X, W, T)
        if newvalue > value:
            value = newvalue
            predictions = tempsolutions.copy()
        if np.all(tempsolutions == int(W.shape[0]-1)):
            break
    return predictions


def compute_brute_value(temp, X, W, T):
    total = np.dot(X[0], W[temp[0]])
    for i in range(1, X.shape[0]):
        total += np.dot(X[i],W[temp[i]]) + T[temp[i-1]][temp[i]]
    return total


if __name__ == '__main__':

    preds_max_sum, objectives = max_sum_decoder()
    print(np.max(objectives))
    print(preds_max_sum)
    with open("decode_output.txt","w") as w:
        for p in preds_max_sum:
            w.write(str(p+1))
            w.write("\n")

    for i,p in enumerate(preds_max_sum):
        print("i=%d : Predicted = %s" % (i+1, chr(p+ord('a'))))

    for i,p in enumerate(brute_force(X[:4], W, Tij)):
        print("i=%d : Brute force predicted = %s" % (i+1, chr(p+ord('a'))))

    # print("Maximum objective score : ", objective.max())
    # print(max_sum_decoder())
