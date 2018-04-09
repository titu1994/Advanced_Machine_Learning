import numpy as np
from multiprocess.pool import Pool

from utils import compute_word_char_accuracy_score, prepare_dataset
from crf_train import matricize_W, matricize_Tij
from crf_evaluate import decode_crf


def process_line(data):
    line, i, X_test, y_test = data

    split = line.split()
    params = np.array(split[1:]).astype(np.float)
    w = matricize_W(params)
    t = matricize_Tij(params)

    predictions = decode_crf(X_test, w, t)

    word_acc, char_acc = compute_word_char_accuracy_score(predictions, y_test)
    print(str(i) + ": ", word_acc)

    return (1. - word_acc)

def save_word_error(infile, outfile, X_test, y_test, l):
    lines = []
    f_vals = []
    # just to keep the file open for as short a time as possible
    file = open(infile, 'r')
    for line in file:
        lines.append(line)
    file.close()

    for i, line in enumerate(lines):
        split = line.split()
        print(str(i) + ": ", end='')
        params = np.array(split[1:]).astype(np.float)
        w = matricize_W(params)
        t = matricize_Tij(params)

        predictions = decode_crf(X_test, w, t)

        word_acc, char_acc = compute_word_char_accuracy_score(predictions, y_test)
        print(word_acc)
        # only word accuracy so just accuracy[0]
        f_vals.append(str(1 - word_acc) + "\n")

    file = open(outfile, 'w')
    file.writelines(f_vals)
    file.close()


def save_word_error_parallel(infile, outfile, X_test, y_test, l):
    lines = []
    # just to keep the file open for as short a time as possible
    file = open(infile, 'r')
    for line in file:
        lines.append(line)
    file.close()

    pool = Pool(4)
    f_vals = []

    data = [(l, i, X_test, y_test) for i, l in enumerate(lines)]
    accuracies = pool.imap(process_line, data)

    pool.close()

    for acc in accuracies:
        f_vals.append(str(acc) + "\n")

    file = open(outfile, 'w')
    file.writelines(f_vals)
    file.close()


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    OPTIMIZERS = ['ADAM', 'SGD']
    LAMBDAS = [1e-4]

    for optm in OPTIMIZERS:
        for lambd in LAMBDAS:
            param_path = 'results/%s_%s.txt' % (optm, lambd)
            loss_path = "results/%s_%s_word_error.txt" % (optm, lambd)

            print("Getting word accuracy data from optimizer %s with lambda %s" % (optm, lambd))
            save_word_error_parallel(param_path, loss_path, X_test, y_test, lambd)

