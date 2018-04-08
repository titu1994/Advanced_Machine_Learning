

import callback_function as cf
import get_data as gd
import full_gradient as fg
import numpy as np

X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")
params = np.zeros(129 * 26 + 26 ** 2)

call_func = cf.callback_function(X_train, y_train, X_test, y_test, "bfgs_1e-2.txt", 0.01)
call_func.delete_file()
opt_params = fg.optimize(params, call_func)
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)

call_func = cf.callback_function(X_train, y_train, X_test, y_test, "bfgs_1e-4.txt", 0.0001)
call_func.delete_file()
opt_params = fg.optimize(params, call_func)
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)



call_func = cf.callback_function(X_train, y_train, X_test, y_test, "bfgs_1e-6.txt", 0.000001)
call_func.delete_file()
opt_params = fg.optimize(params, call_func)
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)