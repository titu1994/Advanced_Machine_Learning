from sklearn.datasets.svmlight_format import load_svmlight_file
import numpy as np
import re
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')

Q1_TRAIN_PATH = "data/decode_input.txt"
Q2_MODEL_PATH = "data/model.txt"
Q2_TRAIN_PATH = "data/train.txt"
train_data = {}

def load_Q1_data():
    with open(Q1_TRAIN_PATH, 'r') as f:
        lines = f.readlines()

    Xi = lines[:100 * 128]
    Wj = lines[100 * 128: 26 * 128 + 100 * 128]
    Tij = lines[26 * 128 + 100 * 128:]

    Xi = np.array(Xi, dtype=np.float32).reshape((100, 128))
    Wj = np.array(Wj, dtype=np.float32).reshape((26, 128))
    Tij = np.array(Tij, dtype=np.float32).reshape((26, 26), order='F')

    print("Xi", Xi.shape, "Wj", Wj.shape, "Tij", Tij.shape)
    return Xi, Wj, Tij

def load_Q2_model():
    with open(Q2_MODEL_PATH, 'r') as f:
        lines = f.readlines()
    Wj = lines[:26*128]
    Tij = lines[26*128:]
    Wj = np.array(Wj, dtype=np.float32).reshape((26, 128))
    Tij = np.array(Tij, dtype=np.float32).reshape((26, 26), order='F')
    print(Wj.shape)
    print(Tij.shape)
    return Wj, Tij
    
def load_Q2_data():
    with open(Q2_TRAIN_PATH, 'r') as f:
        lines = f.readlines()
    for l in lines:
        # get letter
        letter = re.findall(r'[a-z]', l)
        # get all ints
        l = re.findall(r'\d+', l)
        # store letter_id (unique id)
        letter_id = l[0]
        # store next letter id
        next_id = l[1]
        # get current word id
        word_id = l[2]
        # get pos for current letter
        pos = l[3]
        # get pixel values (i, j) for current letter
        p_ij = np.array(l[4:])
        # store letter in dictionary as letter_id -> letter, next_id, word_id, position, pixel_values
        train_data.update({letter_id: [letter, next_id, word_id, pos, p_ij]})
    return train_data

if __name__ == '__main__':
    Xi, Wj, Tij = load_Q1_data()
    Wj_train, Tij_train = load_Q2_model()
    letter_dict = load_Q2_data()
    print(len(letter_dict))

