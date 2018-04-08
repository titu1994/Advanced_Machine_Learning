# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:57:59 2018

@author: Erik
"""
import get_data as gd
import full_gradient as fg

def final_accuracy(file_name, X_train, y_train, X_test, y_test):
    file = open(file_name)
    lines = []
    for line in file:
        lines.append(line)
        
    fin_params = lines[-1].split()[1:]
    fg.print_accuracies(fin_params, X_train, y_train, X_test, y_test)


X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")

print("Final accuracies: Tr.word Tr.letter Ts.word Ts.Letter ")
print("1e-2")
print("BFGS: ", end = '')
final_accuracy("bfgs_1e-2.txt", X_train, y_train, X_test, y_test)
print("SGD : ", end = '')
final_accuracy("sgd_1e-2.txt", X_train, y_train, X_test, y_test)
print()

print("1e-4")
print("BFGS: ", end = '')
final_accuracy("bfgs_1e-4.txt", X_train, y_train, X_test, y_test)
print("SGD : ", end = '')
final_accuracy("sgd_1e-4.txt", X_train, y_train, X_test, y_test)
print()

print("1e-6")
print("BFGS: ", end = '')
final_accuracy("bfgs_1e-6.txt", X_train, y_train, X_test, y_test)
print("SGD : ", end = '')
final_accuracy("sgd_1e-6.txt", X_train, y_train, X_test, y_test)