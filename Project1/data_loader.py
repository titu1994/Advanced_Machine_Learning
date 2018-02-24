from sklearn.datasets.svmlight_format import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')

Q1_TRAIN_PATH = "data/decode_input.txt"


def load_Q1_data():
    with open(Q1_TRAIN_PATH, 'r') as f:
        lines = f.readlines()

    Xi = lines[:100 * 128]
    Wj = lines[100 * 128: 26 * 128 + 100 * 128]
    Tij = lines[26 * 128 + 100 * 128:]

    Xi = np.array(Xi, dtype=np.float64).reshape((100, 128))
    Wj = np.array(Wj, dtype=np.float64).reshape((26, 128))
    Tij = np.array(Tij, dtype=np.float64).reshape((26, 26)).transpose()

    print("Xi", Xi.shape, "Wj", Wj.shape, "Tij", Tij.shape)
    return Xi, Wj, Tij


def plot_Xi(Xi, index):
    xi = Xi[index, :]
    xi = xi.reshape((16, 8))

    plt.imshow(xi, cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    Xi, Wj, Tij = load_Q1_data()

    plot_Xi(Xi, 3)
