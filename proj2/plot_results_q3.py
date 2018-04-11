import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = open('results/ADAM_0.0001_word_error.txt').readlines()
b = open('results/SGD_0.0001_word_error.txt').readlines()
c = open('results/LBFGS_0.0001_word_error.txt').readlines()

c = [x.split() for x in c]
c = c[46:214]

a = pd.DataFrame(a)
b = pd.DataFrame(b)
c = pd.DataFrame(c)

a.columns = ['test_word_err']
a['index'] = range(1, len(a)+1)
b.columns = ['test_word_err']
b['index'] = range(1, len(b)+1)
c.columns = ['iter', 'fn.val', 'gap', 'time', 'feval.num', 'train_lett_err', 'train_word_err', 'test_lett_err', 'test_word_err']

passes = list(pd.to_numeric(a['index']))
fn_val = list(pd.to_numeric(a['test_word_err']))
fn_val = [x*100 for x in fn_val]

passes2 = list(pd.to_numeric(b['index']))
fn_val2 = list(pd.to_numeric(b['test_word_err']))
fn_val2 = [x*100 for x in fn_val2]

passes3 = list(pd.to_numeric(c['iter']))
fn_val3 = list(pd.to_numeric(c['test_word_err']))

plt.plot(passes, fn_val, '-b', label='Adam')
plt.plot(passes2, fn_val2, 'r--', label='SGD')
plt.plot(passes3, fn_val3, 'k', label='LBFGS')

plt.plot()
plt.xlabel('Number of effective pass')
plt.ylabel('Word-wise error')
plt.legend(loc='upper right')
plt.show()