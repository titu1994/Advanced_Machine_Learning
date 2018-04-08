import numpy as np
from proj2.utils import *
from proj2.crf_train import func_to_minimize

def print_function_values(infile, outfile, X_train, y_train, lambd):
    lines = []
    f_vals = []
    # just to keep the file open for as short a time as possible
    file = open(infile, 'r')
    for line in file:
        lines.append(line)
    file.close()

    for line in lines:
        split = line.split()
        print(split[0] + ": ", end = '')
        params = np.array(split[1:]).astype(np.float)
        val = func_to_minimize(params, X_train, y_train, lambd)
        print(val)
        f_vals.append(str(val) + "\n")

    file = open(outfile, 'w')
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
            loss_path = "results/%s_%s_f_evals.txt" % (optm, lambd)

            print("Printing data from optimizer %s with lambda %s" % (optm, lambd))
            print_function_values(param_path, loss_path, X_train, y_train, lambd)

            print()