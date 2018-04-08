import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_results():
	a = open('q2-b.txt').readlines()
	a = [x.split() for x in a]
	a = a[17:189]
	a = pd.DataFrame(a)
	a.columns = ['iter', 'fn.val', 'gap', 'time', 'feval.num', 'train_lett_err', 'train_word_err', 'test_lett_err', 'test_word_err']
	print(a.tail())
	cpu_time = list(pd.to_numeric(a['time']))
	test_lett_err = list(pd.to_numeric(a['test_lett_err']))
	test_word_err = list(pd.to_numeric(a['test_word_err']))
	train_lett_err = list(pd.to_numeric(a['train_lett_err']))
	train_word_err = list(pd.to_numeric(a['train_word_err']))
	# number_of_cores = [8, 6, 4, 2, 1]
	# speedup = [a[3], b[3], c[3], d[3], e[3]]
	# speedup = [float(i) for i in speedup]
	# t1 = speedup[-1]
	# speedup = [t1 / i for i in speedup]
	# print(speedup)
	# print(number_of_cores)
	# plt.plot(number_of_cores, speedup, '-b')
	plt.plot(cpu_time, test_word_err, 'r--', label='Testing')
	plt.plot(cpu_time, train_word_err, '-b', label='Training')
	plt.plot()
	plt.yticks(np.arange(0, max(train_word_err)+10, 10.0))
	plt.xticks(np.arange(0, max(cpu_time), 15.0))
	plt.xlabel('CPU Time (s)')
	plt.ylabel('Word Wise Error')
	plt.legend(loc='upper right')
	plt.show()

read_results()