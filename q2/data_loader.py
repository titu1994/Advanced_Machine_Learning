import csv
import numpy as np
from collections import OrderedDict
import re
import pdb
import sys

Q2_TRAIN_PATH = 'data/train.txt'
Q2_MODEL_PATH = 'data/model.txt'

def load_Q2_model():
	with open(Q2_MODEL_PATH, 'r') as f:
		data = f.readlines()

	Wj = data[:26*128]
	Tij = data[26*128:]
	Wj = np.array(Wj, dtype=np.float32).reshape(26,128)
	Tij = np.array(Tij, dtype=np.float32).reshape((26, 26), order='F')
	return Wj, Tij

def load_Q2_data():
	with open(Q2_TRAIN_PATH, 'r') as f:
		data = f.readlines()
	
	train_data = OrderedDict()

	for l in data:
		letter = re.findall(r'[a-z]', l)
		integers = re.findall(r'\d+', l)
		integers = list(map(int, integers))
		letter_id = integers[0]
		next_id = integers[1]
		word_id = integers[2]
		pos = integers[3]
		pixel_ij = integers[4:]
		if word_id in train_data:
			train_data[word_id].append([letter_id, pos, pixel_ij])
		else:
			train_data[word_id] = [[letter_id, pos, pixel_ij]]

	return train_data

def parse_Q2_data(train_data):
	X_train = np.zeros((25943, 128), dtype=np.float64)
	for i in range(1, len(train_data)):
		for j in range(len(train_data[i])):
			print('i: ' + str(i) + ', j: ' + str(train_data[i][j][2]))

if __name__ == '__main__':
	Wj, Tij = load_Q2_model()
	train_data = load_Q2_data()
	parse_Q2_data(train_data)