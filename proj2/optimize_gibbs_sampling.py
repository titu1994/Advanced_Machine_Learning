import numpy as np
np.random.seed(0)

import random
from proj2.crf_evaluate import decode_crf
from proj2.crf_train import *
from proj2.utils import prepare_dataset, compute_word_char_accuracy_score


"""

NOTE: THIS CODE DOES NOT CURRENTLY WORK

"""
class GibbsSample:

    def __init__(self, lambd, callback_fn):
        self.theta = np.zeros(129 * 26 + 26 ** 2)
        self.learning_rate = 0.01
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.iterations = 37000 # 3700
        self.lambda_param = lambd
        self.callback_fn = callback_fn

    def generate_samples(self, X_train, w, t, samples_per_word):

        # m - length of the word to be sampled
        index = random.randint(0, len(X_train)-1)
        m = len(X_train[index])

        init_y_sample = random.sample(range(26), m)
        samples = []
        numerator = np.zeros(26)
        sample = np.copy(init_y_sample)
        for k in range(samples_per_word):
            rand_char_index = random.randint(0, m-1)

            # for i in range(0, m):
            #     if i != rand_char_index and (i != rand_char_index - 1):
            #         sum = sum + np.dot(w[sample[i]], self.X_train[index][i]) + t[sample[i]][sample[i+1]]

            #initialize a sum array with this sum value

            #for _ in range(100):
            for j in range(26):

                    if rand_char_index == 0:
                        numerator[j] = 100 * np.inner(X_train[index][rand_char_index], w[j]) \
                                       + t[j][sample[rand_char_index + 1]]
                    elif rand_char_index == m - 1:
                        numerator[j] = 100 * t[sample[rand_char_index - 1]][j] \
                                       + np.inner(X_train[index][rand_char_index], w[j])
                    else:
                        numerator[j] = 100 * t[sample[rand_char_index - 1]][j] \
                                       + np.inner(X_train[index][rand_char_index], w[j]) \
                                       + t[j][sample[rand_char_index + 1]]



            best_index = np.argmax(numerator)
            sample[rand_char_index] = best_index

            samples.append(np.copy(sample))

        return (samples, index)


    def train_with_samples(self, X_train):
        # parameter tuning using ADAM

        iteration = 0
        epoch = 0

        m_t = 0  # first order moment
        v_t = 0  # second order moment

        # init params
        theta = np.zeros(129 * 26 + 26 ** 2)

        # init gradient
        grad = np.ones_like(theta)
        while True:
            iteration += 1

            theta_prev = theta

            w = matricize_W(theta)
            t = matricize_Tij(theta)
            samples, sampled_word_index = self.generate_samples(X_train, w, t, 100)#theta[:3354], theta[3354:], 5)

            # calculate gradient
            x_train_array = []
            x_train_array.append(X_train[sampled_word_index])

            y_train_array = []
            y_train_array.append(samples[-1])

            grad = d_optimization_function_per_word(theta, x_train_array, y_train_array, 0, self.lambda_param)
        #   grad = fg.grad_func_word(theta, x_train_array, samples, 0, 0.01)
        #   grad = gc.gradient_avg(theta, (X_train[sampled_word_index], len(samples), 1), samples, len(samples) - 1)

            # biased moments
            m_t = self.beta_1 * m_t + (1 - self.beta_1) * grad
            v_t = self.beta_2 * v_t + (1 - self.beta_2) * np.square(grad)

            # bias corrected first and second order moments
            m_t_hat = m_t / (1 - self.beta_1 ** iteration)
            v_t_hat = v_t / (1 - self.beta_2 ** iteration)

            # update params
            theta = theta_prev - (self.learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + self.epsilon)

            if (iteration % len(X_train)) == 0:
                epoch += 1
                print("Epoch %d : " % (epoch), end='')
                loss_val, avg_grad = self.callback_fn(theta)

                if avg_grad < 2e-6:
                    break


            #print("Iteration %d: Parameters updated" % (iteration))

            #print(theta)
            # termination condition
            if sum(abs(theta - theta_prev)) <= self.epsilon or iteration == self.iterations:
                break

        return theta


if __name__ == '__main__':

    LAMBDA = 1e-2

    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    filepath = "%s_%s.txt" % ('GIBBS', LAMBDA)

    callback = Callback(X_train, y_train, filepath, LAMBDA)

    gibbs  = GibbsSample(LAMBDA, callback_fn=callback.callback_fn_return_vals)
    opt_params = gibbs.train_with_samples(X_train)

    W = matricize_W(opt_params)
    T = matricize_Tij(opt_params)

    y_preds = decode_crf(X_train, W, T)
    word_acc, char_acc = compute_word_char_accuracy_score(y_preds, y_train)
    print("Final train accuracy :", "Word =", word_acc, "Char =", char_acc)

    y_preds = decode_crf(X_test, W, T)
    word_acc, char_acc = compute_word_char_accuracy_score(y_preds, y_test)
    print("Final test accuracy :", "Word =", word_acc, "Char =", char_acc)
