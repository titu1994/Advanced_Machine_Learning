import csv
import numpy as np
import re
import timeit
import pdb
from data_loader import load_Q2_model, load_Q2_data

def forward_pass(Xword, Wj, Tij): # calculate alpha(j, s) # values are not exactly correct...?
	Xword_alpha = np.zeros((len(Xword), 26)) # init (m x 26) matrix to hold alpha values

	Xword_hold = np.zeros((len(Xword), 128)) # init zeros to hold current word pixel values

	for i in range(len(Xword)):
		Xword_hold[i] = Xword[i][2] # XW_hold has pixel values for only letters in current word

	Xword = np.inner(Xword_hold, Wj) # inner product of XW and Wj weights

	for j in range(1, len(Xword)): # for each letter j in XW m
		letter_alpha = Xword_alpha[j-1] + Tij
		letter_alpha_max = np.max(letter_alpha, axis=1)
		letter_alpha = (letter_alpha.T - letter_alpha_max).T
		Xword_alpha[j] = letter_alpha_max + np.log(np.sum(np.exp(letter_alpha + Xword[j-1]), axis=1))

	return Xword_alpha # return alpha values for j=0..m for XW

def backward_pass(Xword, Wj, Tij): # calculate beta(j, s) # values are not exactly correct...?
  	Xword_beta = np.zeros((len(Xword), 26))
  	Tij = Tij.T
  	print('len(Xword): ' + str(len(Xword)))
  	Xword_hold = np.zeros((len(Xword), 128)) # init zeros to hold current word pixel values
	for i in range(len(Xword)):
		Xword_hold[i] = Xword[i][2] # XW_hold has pixel values for only letters in current word

	Xword = np.inner(Xword_hold, Wj) # inner product of XW and Wj weights

  	for j in range(len(Xword) - 2, -1, -1):
  		letter_beta = Xword_beta[j+1] + Tij.T
  		letter_beta_max = np.max(letter_beta, axis=1)
  		letter_beta = (letter_beta.T - letter_beta_max).T
  		Xword_beta[j] = letter_beta_max + np.log(np.sum(np.exp(letter_beta + Xword[j+1]), axis=1))

	return Xword_beta # return beta values for j=m-1..0 for XW

def psi_fn(Xword, y, Tij): # numerator for calculating likelihood value
	final = 0
	for j in range(len(Xword)):
		if j > 0:
			final += Tij[y[j]][y[j-1]]
	return np.exp(total)

def partition_fn(alpha, Xword): # denominator (Z) that normalizes psi_fn
	return np.sum(np.exp(alpha[-1] + Xword[-1]))

def logpy_X(Xword, y, Tij):
	alpha = forward_pass(Xword, Tij)
	return np.log(psi_fn(y, Xword, Tij) / partition_fn(alpha, Xword))

def logpy_X_sum(Wj, Tij, X_train, y, index):
	final = 0
	for j in range(index):
		Xword = np.inner(X_train[j], Wj)
		final += logpy_X(Xword, y[j], Tij)
	return final

if __name__ == '__main__':
	Wj, Tij = load_Q2_model()
	X_train = load_Q2_data()
	forward_pass(X_train[1], Wj, Tij)
	backward_pass(X_train[1], Wj, Tij)