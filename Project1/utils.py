import numpy as np

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