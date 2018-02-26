import numpy as np

def evaluate_structured(f_true, f_pred):
    with open(f_true, 'r') as f_true, open(f_pred, 'r') as f_pred:
        true_char_list = []
        true_word_list = []
        pred_char_list = []
        pred_word_list = []

        prev_word = -1

        for true, pred in zip(f_true, f_pred):
            true_splits = true.split()
            true_char_list.append(true_splits[0])

            true_word = int(true_splits[1][3:])