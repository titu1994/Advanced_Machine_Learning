from sklearn.datasets.svmlight_format import load_svmlight_file
import numpy as np

Q1_TRAIN_PATH = "data/decode_input.txt"


def load_Q1_data():
    with open(Q1_TRAIN_PATH, 'r') as f:
        lines = f.readlines()

    Xi = lines[:100*128]
    Wj = lines[100*128:26*128 + 100*128]
    Tij = lines[26*128 + 100*128:]

    Xi = np.array(Xi, dtype=np.float64).reshape((128, 100)).T
    Wj = np.array(Wj, dtype=np.float64).reshape((128, 26)).T
    Tij = np.array(Tij, dtype=np.float64).reshape((26, 26)).transpose()

    print(Xi.shape, Wj.shape, Tij.shape)
    return Xi, Wj, Tij


if __name__ == '__main__':
    load_Q1_data()
