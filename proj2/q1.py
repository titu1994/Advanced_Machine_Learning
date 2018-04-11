import numpy as np
from utils import *
from crf_train import objective_function

def save_optimization_scores(weight_file, output_filename, X_train, y_train, lambd):
    f_vals = []
    file = open(weight_file, 'r')
    lines = file.readlines()
    file.close()

    for i, line in enumerate(lines):
        splits = line.split()
        print(i, splits[0] + ": ", end = '')

        params = np.array(splits[1:]).astype(np.float)

        val = objective_function(params, X_train, y_train, lambd)

        print(val)
        f_vals.append(str(val) + "\n")

    file = open(output_filename, 'w')
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
            save_optimization_scores(param_path, loss_path, X_train, y_train, lambd)

            print()