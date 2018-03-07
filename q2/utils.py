import numpy as np
import re
import os
from collections import defaultdict

TRANSFORM_PATH = 'data/transform.txt'

if not os.path.exists('result'):
    os.makedirs('result')

def read_data(file_name):
    file = open('data/' + file_name, 'r')
    y = []
    qids = []
    X = []

    for line in file:
        temp = line.split()
        # get label
        label_string = temp[0]
        # normalizes y to 0...25 instead of 1...26
        y.append(int(label_string) - 1)

        # get qid
        qid_string = re.split(':', temp[1])[1]
        qids.append(int(qid_string))

        # get x values
        x = np.zeros(128)

        # do we need a 1 vector?
        # x[128] = 1
        for elt in temp[2:]:
            index = re.split(':', elt)
            x[int(index[0]) - 1] = 1
        X.append(x)
    y = np.array(y)
    qids = np.array(qids)
    return y, qids, X


def read_data_formatted(file_name):
    # get initial output
    y, qids, X = read_data(file_name)
    y_tot = []
    X_tot = []
    current = 0

    y_tot.append([])
    X_tot.append([])

    for i in range(len(y)):
        y_tot[current].append(y[i])
        X_tot[current].append(X[i])

        if (i + 1 < len(y) and qids[i] != qids[i + 1]):
            y_tot[current] = np.array(y_tot[current])
            X_tot[current] = np.array(X_tot[current])
            y_tot.append([])
            X_tot.append([])
            current = current + 1

    return X_tot, y_tot


# parameter set for 2a
def get_params():
    file = open('data/model.txt', 'r')
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)


def flatten_dataset(X):
    x = []
    for w_list in X:
        for w in w_list:
            x.append(w)

    x = np.array(x)
    return x

def reshape_dataset(X, ref_y):
    x = [[]]
    index = 0
    for i, words in enumerate(ref_y):
        count = len(words)

        for j in range(count):
            x[i].append(X[index])
            index += 1

        x.append([])

    return x


def compute_accuracy(y_preds, y_true):
    word_count = 0
    correct_word_count = 0
    letter_count = 0
    correct_letter_count = 0

    for y_t, y_p in zip(y_true, y_preds):
        word_count += 1
        if np.array_equal(y_t, y_p):
            correct_word_count += 1

        letter_count += len(y_p)
        correct_letter_count += np.sum(y_p == y_t)

    return correct_word_count / word_count, correct_letter_count / letter_count


def evaluate_structured(f_true, f_pred):
    with open(f_true, 'r') as f_true, open(f_pred, 'r') as f_pred:
        true_char_list = []
        true_word_list = []
        pred_char_list = []
        pred_word_list = []

        prev_word = -100

        for true, pred in zip(f_true, f_pred):
            true_splits = true.split()
            true_char = int(true_splits[0])
            true_char_list.append(true_char)

            pred_char = int(pred)
            if hasattr(pred_char, 'len') > 0:
                pred_char = pred_char[0]

            pred_char_list.append(pred_char)

            true_word = int(true_splits[1][4:])
            if true_word == prev_word:
                true_word_list[-1].append(true_char)
                pred_word_list[-1].append(pred_char)
            else:
                true_word_list.append([true_char])
                pred_word_list.append([pred_char])
                prev_word = true_word

        char_correct_count = 0
        word_correct_count = 0

        for true_char, pred_char in zip(true_char_list, pred_char_list):
            if true_char == pred_char:
                char_correct_count += 1

        for true_word, pred_word in zip(true_word_list, pred_word_list):
            if np.array_equal(true_word, pred_word):
                word_correct_count += 1

        char_acc = float(char_correct_count) / float(len(true_char_list))
        word_acc = float(word_correct_count) / float(len(true_word_list))

        print("Character level accuracy : %0.4f (%d / %d)" % (char_acc, char_correct_count, len(true_char_list)))
        print("Word level accuracy : %0.4f (%d / %d)" % (word_acc, word_correct_count, len(true_word_list)))

        return char_acc, word_acc

def _rotate(Xi, alpha):
    Xi = Xi.reshape((16, 8))
    alpha = float(alpha)

    y = imrotate(Xi, angle=alpha)

    x_height, x_width = Xi.shape
    y_height, y_width = y.shape

    from_x = int((y_height + 1 - x_height) // 2)
    from_y = int((y_width + 1 - x_width) // 2)

    y = y[from_x:from_x + x_height, from_y: from_y + x_width]

    idx = np.where(y == 0)
    y[idx] = Xi[idx]

    return y


def _translation(Xi, offsets):
    Xi = Xi.reshape((16, 8))
    x_height, x_width = Xi.shape

    x_offset, y_offset = offsets
    x_offset, y_offset = int(x_offset), int(y_offset)

    y = Xi

    y[max(0, x_offset): min(x_height, x_height + x_offset),
      max(0, y_offset): min(x_width, x_width + y_offset)] = Xi[max(0, 1 - x_offset): min(x_height, x_height - x_offset),
                                                               max(0, 1 - y_offset): min(x_width, x_width - y_offset)]

    y[x_offset: x_height, y_offset: x_width] = Xi[0: x_height - x_offset, 0: x_width - y_offset]

    return y


def transform_dataset(train_set, limit, word_ids):
    if limit == 0:
        return train_set

    # build an inverse word dictionary
    # word_dict = defaultdict(list)

    # for key, value in train_set.items():
        # word_id = value[2]
        # word_dict[word_id].append(key)

    with open(TRANSFORM_PATH, 'r') as f:
        lines = f.readlines()

    lines = lines[:limit]

    for line in lines:
        splits = line.split()
        action = splits[0]
        target_word = int(splits[1])
        # args = int(splits[2:])
        args = list(map(int, splits[2:]))

        # get all of the ids in train set which have this word in them
        target_image_ids = word_ids[target_word]

        for image_id in target_image_ids:
            value_set = train_set[image_id]
            image = value_set[-1]

            if action == 'r':
                alpha = args[0]
                image = _rotate(image, alpha)
            else:
                offsets = args
                image = _translation(image, offsets)

            value_set[-1] = image.flatten()

            train_set[image_id] = value_set

    return train_set

#
# def evaluate_crf(y_true, y_preds, word_ids):
#     true_word_list = []
#     pred_word_list = []
#
#     prev_word = -100
#
#     for i, (true, pred) in enumerate(zip(y_true, y_preds)):
#         true_word = int(word_ids[i])
#         if true_word == prev_word:
#             true_word_list[-1].append(true)
#             pred_word_list[-1].append(pred)
#         else:
#             true_word_list.append([true])
#             pred_word_list.append([pred])
#             prev_word = true_word
#
#     char_correct_count = 0
#     word_correct_count = 0
#
#     for true_char, pred_char in zip(y_true, y_preds):
#         if true_char == pred_char:
#             char_correct_count += 1
#
#     for true_word, pred_word in zip(true_word_list, pred_word_list):
#         if np.array_equal(true_word, pred_word):
#             word_correct_count += 1
#
#     char_acc = (char_correct_count) / float(y_true.shape[0])
#     word_acc = float(word_correct_count) / float(len(true_word_list))
#
#     print("Character level accuracy : %0.4f (%d / %d)" % (char_acc, char_correct_count, y_true.shape[0]))
#     print("Word level accuracy : %0.4f (%d / %d)" % (word_acc, word_correct_count, len(true_word_list)))
#
#     return char_acc, word_acc
