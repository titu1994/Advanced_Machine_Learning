import numpy as np
from scipy.misc import imrotate
from collections import defaultdict

TRANSFORM_PATH = "data/transform.txt"


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


def evaluate_crf(y_true, y_preds, word_ids):
    true_word_list = []
    pred_word_list = []

    prev_word = -100

    for i, (true, pred) in enumerate(zip(y_true, y_preds)):
        true_word = int(word_ids[i])
        if true_word == prev_word:
            true_word_list[-1].append(true)
            pred_word_list[-1].append(pred)
        else:
            true_word_list.append([true])
            pred_word_list.append([pred])
            prev_word = true_word

    char_correct_count = 0
    word_correct_count = 0

    for true_char, pred_char in zip(y_true, y_preds):
        if true_char == pred_char:
            char_correct_count += 1

    for true_word, pred_word in zip(true_word_list, pred_word_list):
        if np.array_equal(true_word, pred_word):
            word_correct_count += 1

    char_acc = (char_correct_count) / float(y_true.shape[0])
    word_acc = float(word_correct_count) / float(len(true_word_list))

    print("Character level accuracy : %0.4f (%d / %d)" % (char_acc, char_correct_count, y_true.shape[0]))
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


def transform_dataset(train_set, limit):
    if limit == 0:
        return train_set

    # build an inverse word dictionary
    word_dict = defaultdict(list)

    for key, value in train_set.items():
        word_id = value[2]
        word_dict[word_id].append(key)

    with open(TRANSFORM_PATH, 'r') as f:
        lines = f.readlines()

    lines = lines[:limit]

    for line in lines:
        splits = line.split()
        action = splits[0]
        target_word = splits[1]
        args = splits[2:]

        # get all of the ids in train set which have this word in them
        target_image_ids = word_dict[target_word]

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

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
