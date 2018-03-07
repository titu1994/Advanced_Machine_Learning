import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf.python.ops.crf import crf_decode, crf_log_likelihood
from tensorflow.contrib.opt.python.training.external_optimizer import ScipyOptimizerInterface

np.random.seed(0)
tf.set_random_seed(0)

from Project1.data_loader import load_Q2_data, calculate_word_lengths, prepare_dataset

RESTORE_CHECKPOINT = False
C = 1000

train_dataset, test_dataset = load_Q2_data()
train_word_lengths = calculate_word_lengths(train_dataset)
test_word_lengths = calculate_word_lengths(test_dataset)

X_train, y_train = prepare_dataset(train_dataset, train_word_lengths)
X_test, y_test = prepare_dataset(test_dataset, test_word_lengths)

print("Train shape : ", X_train.shape, y_train.shape)
print("Test shape : ", X_test.shape, y_test.shape)

num_train_examples = len(train_word_lengths)
num_test_examples = len(test_word_lengths)

num_train_words = X_train.shape[1]  # random
num_test_words = X_test.shape[1]  # random

num_features = X_train.shape[2]  # 128
num_tags = 26  # 26

with tf.Session() as sess:
    x_t = tf.constant(X_train, dtype=tf.float32, name='X_train')
    y_t = tf.constant(y_train, dtype=tf.int32, name='y_train')

    x_test_t = tf.constant(X_test, dtype=tf.float32, name='X_test')
    y_test_t = tf.constant(y_test, dtype=tf.int32, name='y_test')

    train_sequence_lengths_t = tf.constant(train_word_lengths, dtype=tf.int32, name='train_sequence_lengths')
    test_sequence_lengths_t = tf.constant(test_word_lengths, dtype=tf.int32, name='test_sequence_lengths')

    Cs = [1.0, 10.0, 100.0, 1000.0]

    w_t = tf.get_variable('W', shape=(num_features, num_tags), dtype=tf.float32,
                          regularizer=None, initializer=tf.initializers.zeros())

    transition_weights_t = tf.get_variable('T', shape=(num_tags, num_tags), dtype=tf.float32,
                                           regularizer=None, initializer=tf.initializers.zeros())

    saver = tf.train.Saver(max_to_keep=1)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    x_t_features = tf.reshape(x_t, [-1, num_features], name='X_flattened')

    if RESTORE_CHECKPOINT:
        ckpt_path = tf.train.latest_checkpoint('models/')

        if ckpt_path is not None and tf.train.checkpoint_exists(ckpt_path):
            print("Loading Encoder Checkpoint !")
            saver.restore(sess, ckpt_path)

    for C in Cs:
        sess.run(tf.global_variables_initializer())

        scores = tf.matmul(x_t_features, w_t, name='energies')
        scores = tf.reshape(scores, [num_train_examples, num_train_words, num_tags])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        log_likelihood, transition_weights_t = crf_log_likelihood(scores, y_t, train_sequence_lengths_t, transition_weights_t)

        x_train_t_features = tf.reshape(x_t, [-1, num_features], name='X_train_flattened')
        x_test_t_features = tf.reshape(x_test_t, [-1, num_features], name='X_test_flattened')

        test_scores = tf.matmul(x_test_t_features, w_t, name='test_energies')
        test_scores = tf.reshape(test_scores, [num_test_examples, num_test_words, num_tags])

        # Compute the viterbi sequence and score.
        viterbi_sequence_train, viterbi_train_scores = crf_decode(scores, transition_weights_t, train_sequence_lengths_t)
        viterbi_sequence, viterbi_score = crf_decode(test_scores, transition_weights_t, test_sequence_lengths_t)

        # Add a training op to tune the parameters.
        loss = -C * tf.reduce_mean(log_likelihood)
        loss += tf.nn.l2_loss(w_t)
        loss += 0.5 * tf.reduce_sum(tf.square(transition_weights_t))

        optimizer = ScipyOptimizerInterface(loss)

        # Train for a fixed number of iterations.
        optimizer.minimize(sess)

        mask = (np.expand_dims(np.arange(num_train_words), axis=0) <
                np.expand_dims(train_word_lengths, axis=1))
        total_labels = np.sum(train_word_lengths)

        tf_viterbi_sequence, logloss = sess.run([viterbi_sequence_train, loss])

        correct_labels = np.sum((y_train == tf_viterbi_sequence) * mask)
        accuracy = 100.0 * correct_labels / float(total_labels)

        print("\nC = %d" % (C))
        print("Train | Loss : %0.4f | Accuracy: %.2f%%" % (logloss, accuracy))

        mask = (np.expand_dims(np.arange(num_test_words), axis=0) <
                np.expand_dims(test_word_lengths, axis=1))
        total_labels = np.sum(test_word_lengths)

        tf_viterbi_sequence = sess.run(viterbi_sequence)

        correct_labels = np.sum((y_test == tf_viterbi_sequence) * mask)
        accuracy = 100.0 * correct_labels / float(total_labels)

        print("Test | Accuracy: %.2f%%" % (accuracy))

        saver.save(sess, 'models/crf.ckpt', global_step)


