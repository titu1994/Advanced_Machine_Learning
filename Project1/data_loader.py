from collections import OrderedDict
import numpy as np
import re
import matplotlib.pyplot as plt

# plt.style.use('seaborn-paper')

Q1_TRAIN_PATH = "data/decode_input.txt"
Q2_MODEL_PATH = "data/model.txt"
Q2_TRAIN_PATH = "data/train.txt"
Q2_TEST_PATH = "data/test.txt"

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
    Wj = np.array(Wj, dtype=np.float32).reshape((128, 26)).T
    Tij = np.array(Tij, dtype=np.float32).reshape((26, 26), order='F')
    print(Wj.shape)
    print(Tij.shape)
    return Wj, Tij
    
def load_Q2_data():
    train_data = OrderedDict()

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
        p_ij = np.array(l[4:], dtype=np.float32)
        # store letter in dictionary as letter_id -> letter, next_id, word_id, position, pixel_values
        train_data[letter_id] = [letter, next_id, word_id, pos, p_ij]

    test_data = OrderedDict()

    with open(Q2_TEST_PATH, 'r') as f:
        lines = f.readlines()

    for l in lines:
        letter = re.findall(r'[a-z]', l)
        l = re.findall(r'\d+', l)
        letter_id = l[0]
        next_id = l[1]
        word_id = l[2]
        pos = l[3]
        p_ij = np.array(l[4:], dtype=np.float32)
        # store letter in dictionary as letter_id -> letter, next_id, word_id, position, pixel_values
        test_data[letter_id] = [letter, next_id, word_id, pos, p_ij]

    return train_data, test_data

def calculate_word_lengths(dataset):
    word_list = []
    prev_word = -100

    for i, (key, value) in enumerate(dataset.items()):
        letter, next_id, word_id, pos, p_ij = value

        if word_id == prev_word:
            word_list[-1] += 1
        else:
            word_list.append(1)
            prev_word = word_id

    return word_list

def prepare_dataset(dataset, word_length_list):
    max_length = max(word_length_list)
    num_samples = len(word_length_list)

    X = np.zeros((num_samples, max_length, 128), dtype='float32')
    y = np.zeros((num_samples, max_length), dtype='int32')

    dataset_pointer = 0
    dataset_values = list(dataset.values())

    for i in range(num_samples):
        num_words = word_length_list[i]

        for j in range(num_words):
            letter, next_id, word_id, pos, p_ij = dataset_values[j + dataset_pointer]
            X[i, j, :] = p_ij

            letter_to_int = int(ord(letter[0]) - ord('a'))
            y[i, j] = letter_to_int

        dataset_pointer += num_words

    return X, y


if __name__ == '__main__':
    #Xi, Wj, Tij = load_Q1_data()
    #Wj_train, Tij_train = load_Q2_model()
    train_set, test_set = load_Q2_data()
    train_word_list = calculate_word_lengths(train_set)

    print(max(train_word_list))


