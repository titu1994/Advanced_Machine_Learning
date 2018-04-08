# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 15:59:43 2018

@author: Erik
"""
import numpy as np
import full_gradient as fg
import get_data as gd
import gradient_calculation as gc
import time

def print_word_error(infile, outfile, X_test, y_test, l):
    lines = []
    f_vals = []
    #just to keep the file open for as short a time as possible
    file = open(infile, 'r')
    for line in file:
        lines.append(line)
    file.close()
    for i, line in enumerate(lines):
        split = line.split()
        print(str(i) + ": ", end = '')
        params = np.array(split[1:]).astype(np.float)
        w = gc.w_matrix(params)
        t = gc.t_matrix(params)


        predictions = fg.predict(X_test, w, t)
        
        accuracy = fg.accuracy(predictions, y_test)
        print(accuracy[0])
        #only word accuracy so just accuracy[0]
        f_vals.append(str(1 - accuracy[0]) + "\n")
        
    file = open(outfile, 'w')
    file.writelines(f_vals)
    file.close()
    
X_test, y_test = gd.read_data("test_sgd.txt")

'''
start = time.time()
print_word_error("bfgs_1e-4.txt", "bfgs_1e-4_word_error.txt", X_test, y_test, 0.0001)
print("time for first iteration %3f" %(time.time() - start))
'''

print_word_error("sgd_1e-4.txt", "sgd_1e-4_word_error.txt", X_test, y_test, 0.0001)


'''todo: same stuff with adam when it's done'''