# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:40:41 2018

@author: Erik
"""

import callback_function as cf
import get_data as gd

X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")

print("sgd 1e-2 final accuracy")
sgd_2 = cf.callback_function(X_train, y_train, X_test, y_test, "sgd_1e-2.txt")
sgd_2.print_final_accuracy()

print("sgd 1e-4 final accuracy")
sgd_4 = cf.callback_function(X_train, y_train, X_test, y_test, "sgd_1e-4.txt")
sgd_4.print_final_accuracy()

print("sgd 1e-6 final accuracy")
sgd_6 = cf.callback_function(X_train, y_train, X_test, y_test, "sgd_1e-6.txt")
sgd_6.print_final_accuracy()

print("bfgs 1e-2 final accuracy")
bfgs_2 = cf.callback_function(X_train, y_train, X_test, y_test, "bfgs_1e-2.txt")
bfgs_2.print_final_accuracy()

print("bfgs 1e-4 final accuracy")
bfgs_4 = cf.callback_function(X_train, y_train, X_test, y_test, "bfgs_1e-4.txt")
bfgs_4.print_final_accuracy()

print("bfgs 1e-6 final accuracy")
bfgs_6 = cf.callback_function(X_train, y_train, X_test, y_test, "bfgs_1e-6.txt")
bfgs_6.print_final_accuracy()