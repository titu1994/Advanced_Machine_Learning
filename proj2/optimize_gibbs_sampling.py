import numpy as np
np.random.seed(0)

from crf_evaluate import decode_crf
from crf_train import *
from utils import prepare_dataset, compute_word_char_accuracy_score


"""

NOTE: THIS CODE DOES NOT CURRENTLY WORK

"""
class SamplingOptimizer:

    def __init__(self, lambd, callback_fn):
        self.parameters = np.zeros(129 * 26 + 26 ** 2)
        self.learning_rate = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.iterations = 37000 # 3700
        self.lambda_param = lambd
        self.callback_fn = callback_fn

    def sample_dataset(self, X_train, W, T, num_samples):

        index = np.random.randint(0, len(X_train)-1)
        m = len(X_train[index])

        initial_sample_letters = np.random.randint(0, 26, m)
        samples = []
        numerator = np.zeros(26)
        sample = np.copy(initial_sample_letters)

        for k in range(num_samples):
            rand_char_index = np.random.randint(0, m-1)

            for j in range(26):
                energies = np.inner(X_train[index][rand_char_index], W[j])
                if rand_char_index == 0:
                    numerator[j] = energies + T[j][sample[rand_char_index + 1]]
                elif rand_char_index == m - 1:
                    numerator[j] = energies + T[sample[rand_char_index - 1]][j]
                else:
                    numerator[j] = energies + T[sample[rand_char_index - 1]][j] + T[j][sample[rand_char_index + 1]]

            max_numerator = np.argmax(numerator)
            sample[rand_char_index] = max_numerator

            samples.append(np.copy(sample))

        return (samples, index)


    def train(self, X_train):

        iteration = 0
        epoch = 0

        M = 0  # first order moment
        V = 0  # second order moment

        parameters = np.zeros(129 * 26 + 26 ** 2)

        while True:
            iteration += 1

            previous_params = parameters

            W = matricize_W(parameters)
            T = matricize_Tij(parameters)

            samples, sampled_word_index = self.sample_dataset(X_train, W, T, num_samples=100)

            # calculate gradient
            x_train_array = [X_train[sampled_word_index]]
            y_train_array = [samples[-1]]

            grad = d_optimization_function_per_word(parameters, x_train_array, y_train_array, 0, self.lambda_param)

            # biased moments
            M = self.beta1 * M + (1 - self.beta1) * grad
            V = self.beta2 * V + (1 - self.beta2) * np.square(grad)

            # bias corrected first and second order moments
            m_t_hat = M / (1 - self.beta1 ** iteration)
            v_t_hat = V / (1 - self.beta2 ** iteration)

            # update params
            parameters = previous_params - (self.learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + self.epsilon)

            if (iteration % len(X_train)) == 0:
                epoch += 1

                print("Epoch %d : " % (epoch), end='')
                loss_val, avg_grad = self.callback_fn(parameters)

                if avg_grad < 2e-6:
                    break

            if sum(abs(parameters - previous_params)) <= self.epsilon or iteration == self.iterations:
                break

        return parameters


if __name__ == '__main__':

    LAMBDA = 1e-2

    X_train, y_train = prepare_dataset("train_sgd.txt")
    X_test, y_test = prepare_dataset("test_sgd.txt")

    filepath = "%s_%s.txt" % ('GIBBS', LAMBDA)

    callback = Callback(X_train, y_train, filepath, LAMBDA)

    gibbs  = SamplingOptimizer(LAMBDA, callback_fn=callback.callback_fn_return_vals)
    opt_params = gibbs.train(X_train)

    W = matricize_W(opt_params)
    T = matricize_Tij(opt_params)

    y_preds = decode_crf(X_train, W, T)
    word_acc, char_acc = compute_word_char_accuracy_score(y_preds, y_train)
    print("Final train accuracy :", "Word =", word_acc, "Char =", char_acc)

    y_preds = decode_crf(X_test, W, T)
    word_acc, char_acc = compute_word_char_accuracy_score(y_preds, y_test)
    print("Final test accuracy :", "Word =", word_acc, "Char =", char_acc)
