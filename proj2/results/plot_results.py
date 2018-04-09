import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = open('ADAM_1e-06_f_evals.txt').readlines()
b = open('SGD_1e-06_f_evals.txt').readlines()
c = open('LAMBDA_1e-06_f_evals.txt').readlines()
a = [x.split() for x in a]
b = [x.split() for x in b]
c = [x.split() for x in c]
c = c[46:297]
a = pd.DataFrame(a)
b = pd.DataFrame(b)
c = pd.DataFrame(c)
a.columns = ['passes', 'fn_val']
b.columns = ['passes', 'fn_val']
c.columns = ['iter', 'fn.val', 'gap', 'time', 'feval.num', 'train_lett_err', 'train_word_err', 'test_lett_err', 'test_word_err']

passes = list(pd.to_numeric(a['passes']))
fn_val = list(pd.to_numeric(a['fn_val']))
passes2 = list(pd.to_numeric(b['passes']))
fn_val2 = list(pd.to_numeric(b['fn_val']))
passes3 = list(pd.to_numeric(c['iter']))
fn_val3 = list(pd.to_numeric(c['fn.val']))

#plt.plot(passes, fn_val, '-b', label='Adam')
#plt.plot(passes2, fn_val2, 'r--', label='SGD')
plt.plot(passes3, fn_val3, 'k', label='LBFGS')
plt.plot()
plt.xlabel('Effective number of passes')
plt.ylabel('Function value')
plt.legend(loc='upper right')
plt.show()