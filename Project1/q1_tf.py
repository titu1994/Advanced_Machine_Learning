import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf.python.ops.crf import crf_decode
np.random.seed(0)
tf.set_random_seed(0)

from data_loader import load_Q1_data

X, W, Tij = load_Q1_data()

num_examples = 1  # 1 sample_dataset
num_words = X.shape[0]  # 100
num_features = X.shape[1]  # 128
num_tags = W.shape[0]  # 26

X = X.reshape((1, num_words, num_features)).astype(np.float32)

with tf.Session() as sess:
    x_t = tf.constant(X, dtype=tf.float32, name='X')
    w_t = tf.constant(W, dtype=tf.float32, name='W')
    t_t = tf.constant(Tij, dtype=tf.float32, name='T')
    sequence_lengths_t = tf.constant([num_words], dtype=tf.int32)

    x_t_features = tf.reshape(x_t, [-1, num_features], name='X_flattened')
    scores = tf.matmul(x_t_features, w_t, transpose_b=True, name='energies')
    scores = tf.reshape(scores, [num_examples, num_words, num_tags])

    viterbi_sequence, viterbi_score = crf_decode(scores, t_t, sequence_lengths_t)

    sequence = sess.run(viterbi_sequence)[0]
    sequence = [chr(s + 65) for s in sequence]

    for i, s in enumerate(sequence):
        print("i=%d : Predicted = %s" % (i + 1, s))




