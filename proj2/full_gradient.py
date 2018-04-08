

import gradient_calculation as gc
from scipy.optimize import fmin_bfgs
from time import time
import numpy as np
import decode as dc
 

def func_to_minimize(params, X_train, y_train, l):
    num_examples = len(X_train)
    reg = 1/2 * np.sum(params ** 2)
    avg_prob = gc.log_p_y_given_x_avg(params, X_train, y_train, num_examples)
    return -avg_prob + l * reg

def func_to_minimize_word(params, X_train, y_train, word_num, l):
    w = gc.w_matrix(params)
    t = gc.t_matrix(params)
    reg = 1/2 * np.sum(params ** 2)
    w_x = np.inner(X_train[word_num], w)
    prob = gc.log_p_y_given_x(w_x, y_train[word_num], t, word_num)
    
    return -prob + l * reg

def grad_func(params, X_train, y_train, l):
    num_examples = len(X_train)
    grad_avg =  gc.gradient_avg(params, X_train, y_train, num_examples)
    grad_reg = params
    return - grad_avg + l * grad_reg

def grad_func_word(params, X_train, y_train, word_num, l):
    w = gc.w_matrix(params)
    t = gc.t_matrix(params)
    grad_avg =  gc.gradient_word(X_train, y_train, w, t, word_num)
    grad_reg = params
    return - grad_avg + l * grad_reg

    

def optimize(params, call_func):
    start = time()
    opt_params = fmin_bfgs(func_to_minimize, params, grad_func, 
                           (call_func.X_train, call_func.y_train, call_func.lmda), 
                           maxiter = 100, 
                           gtol = 1e-30, 
                           callback = call_func.call_back)
    print("Total time: ", end = '')
    print(time() - start)
    return opt_params
    
def accuracy(y_pred, y_act):
    word_count = 0
    correct_word_count = 0
    letter_count = 0
    correct_letter_count = 0
    for i in range(len(y_pred)):
        word_count += 1
        correct_word_count += np.sum(y_pred[i] == y_act[i]) == len(y_pred[i])
        letter_count += len(y_pred[i])
        correct_letter_count += np.sum(y_pred[i] == y_act[i])
    return correct_word_count/word_count, correct_letter_count/letter_count

def get_optimal_params(name):
    
    file = open('../result/' + name + '.txt', 'r') 
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)

def predict(X, w, t):
    y_pred = []
    for i, x in enumerate(X):
        M = dc.decode(x, w, t)
        
        y_pred.append(dc.get_solution_from_M(M, x, w, t))
    return y_pred

def print_accuracies(params, X_train, y_train, X_test, y_test):
    w = gc.w_matrix(params)
    t = gc.t_matrix(params)
    predictions = predict(X_train, w, t)
    train_accuracy = accuracy(predictions, y_train)
    print("%.3f, " %train_accuracy[0], end = '')
    print("%.3f, " %train_accuracy[1], end = '')
    
    predictions = predict(X_test, w, t)
    test_accuracy = accuracy(predictions, y_test)
    print("%.3f, " %test_accuracy[0], end = '')
    print("%.3f " %test_accuracy[1])