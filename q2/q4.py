from subprocess import run, PIPE
import os
import matplotlib.pyplot as plt
import sys

import numpy as np
from sklearn.svm import LinearSVC
from utils import flatten_dataset, reshape_dataset, read_data_formatted
from q3 import train_evaluate_linear_svm, plot_scores

CHAR_CV_SCORES = []
WORD_CV_SCORES = []

struct_train_path = "train_struct.txt"
struct_test_path = "test_struct.txt"

def Q4():
    limits = [0, 500, 1000, 1500, 2000]  # [1.0, 10.0, 100.0, 1000.0]
    for limit in limits:
        train_evaluate_linear_svm(C=1.0, transform_trainset=True, limit=limit)

    plot_scores(limits, scale=None, xlabel='distortion count')

if __name__ == '__main__':
	Q4()