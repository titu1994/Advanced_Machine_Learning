import time
from multiprocess.pool import Pool

from utils import prepare_dataset
from q3 import process_line


def print_word_error_parallel(results_file, X_test, y_test, pool, optimizer, lambd):
    file = open(results_file, 'r')
    lines = file.readlines()
    file.close()

    # only need the last step of training to evaluate
    lines = [lines[-1]]

    # packs a tuple - line, index, X_test, y_test, verbose (should print or not)
    data = [(l, i, X_test, y_test, False) for i, l in enumerate(lines)]
    accuracies = pool.imap(process_line, data)  # maintains order when processing in parallel

    return accuracies, optimizer, lambd


if __name__ == '__main__':
    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    OPTIMIZERS = ['BFGS', 'ADAM', 'SGD']
    LAMBDAS = [1e-2, 1e-4, 1e-6]

    # 9 possible combinations, process them all at once.
    # Requires a ton of memory and CPU compute (which can hang laptops with < 4 cores),  so reduce it if needed
    pool = Pool(9)
    result_list = []

    t1 = time.time()
    for optm in OPTIMIZERS:
        for lambd in LAMBDAS:
            param_path = 'results/%s_%s.txt' % (optm, lambd)
            print("Getting word accuracy data from optimizer %s with lambda %s" % (optm, lambd))

            # parallel processing, takes a lot of memory
            result = print_word_error_parallel(param_path, X_test, y_test, pool, optm, lambd)
            result_list.append(result)

    pool.close()

    print('\n', '*' * 80, '\n')

    for result in result_list:
        result_holder, model, lambd = result
        result = next(result_holder)

        print("Model = %s | Lambda = %0.6f | Word Accuracy = %f" % (model, lambd, result))

    print()
    print("Time taken : ", time.time() - t1, "seconds")
