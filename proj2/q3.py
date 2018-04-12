import numpy as np
from multiprocess.pool import Pool

from utils import compute_word_char_accuracy_score, prepare_dataset
from crf_train import matricize_W, matricize_Tij
from crf_evaluate import decode_crf


def process_line(packed_line):
    line, i, X_test, y_test, verbose = packed_line

    split = line.split()
    params = np.array(split[1:]).astype(np.float)
    w = matricize_W(params)
    t = matricize_Tij(params)

    predictions = decode_crf(X_test, w, t)

    word_acc, char_acc = compute_word_char_accuracy_score(predictions, y_test)

    if verbose:
        print(str(i) + ": ", word_acc)

    return (1. - word_acc)


def save_word_error(result_file, output_file, X_test, y_test):
    f_vals = []

    file = open(result_file, 'r')
    lines = file.readlines()
    file.close()

    for i, line in enumerate(lines):
        split = line.split()
        print(i, ": ", end='')
        params = np.array(split[1:]).astype(np.float)
        w = matricize_W(params)
        t = matricize_Tij(params)

        predictions = decode_crf(X_test, w, t)

        word_acc, char_acc = compute_word_char_accuracy_score(predictions, y_test)
        print(word_acc)
        # only word accuracy so just accuracy[0]
        f_vals.append(str(1 - word_acc) + "\n")

    file = open(output_file, 'w')
    file.writelines(f_vals)
    file.close()


def save_word_error_parallel(results_file, output_file, X_test, y_test):
    file = open(results_file, 'r')
    lines = file.readlines()
    file.close()

    pool = Pool(4)  # uses up 4 cores per call (for 8 core machine), so change this to match half your cpu count
    f_vals = []

    # packs a tuple - line, index, X_test, y_test, verbose (should print or not)
    data = [(l, i, X_test, y_test, True) for i, l in enumerate(lines)]

    # process it in correct order
    accuracies = pool.imap(process_line, data)  # maintains order when processing in parallel

    # wait while processing the data
    pool.close()

    for acc in accuracies:
        f_vals.append(str(acc) + "\n")

    file = open(output_file, 'w')
    file.writelines(f_vals)
    file.close()


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    OPTIMIZERS = ['BFGS', 'ADAM', 'SGD']
    LAMBDAS = [1e-2, 1e-4, 1e-6]

    for optm in OPTIMIZERS:
        for lambd in LAMBDAS:
            param_path = 'results/%s_%s.txt' % (optm, lambd)
            loss_path = "results/%s_%s_word_error.txt" % (optm, lambd)

            print("Getting word accuracy data from optimizer %s with lambda %s" % (optm, lambd))

            # sequential processing, takes a lot of time
            # save_word_error(param_path, loss_path, X_test, y_test)

            # parallel processing, takes a lot of memory
            save_word_error_parallel(param_path, loss_path, X_test, y_test)

