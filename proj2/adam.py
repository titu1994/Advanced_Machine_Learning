import numpy as np
import full_gradient as fg
import random as rd
import get_data as gd
import callback_function as cf

class Adam:

    # params with default values
    def __init__(self, iterations, lambda_param=0.000001, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):

        self.lambda_param = lambda_param
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.iterations = iterations

    # Reference :  https://arxiv.org/pdf/1412.6980.pdf
    def update(self, call_func):

        epoch = 0
        iter = 0
        m_t = 0         # first order moment
        v_t = 0         # second order moment

        # init params
        theta = np.ones(129 * 26 + 26 ** 2)

        # init gradient
        grad = np.ones_like(theta)

        while True:
            epoch += 1

            theta_prev = theta

            #calculate gradient
            grad = fg.grad_func_word(theta, call_func.X_train, call_func.y_train, rd.randint(0, len(X_train) - 1), self.lambda_param)

            # biased moments
            m_t = self.beta_1 * m_t + (1 - self.beta_1) * grad
            v_t = self.beta_2 * v_t + (1-self.beta_2) * np.square(grad)

            # bias corrected first and second order moments
            m_t_hat = m_t / (1 - self.beta_1 ** epoch)
            v_t_hat = v_t / (1 - self.beta_2 ** epoch)

            # update params
            theta = theta_prev - (self.learning_rate * m_t_hat) / (np.sqrt(v_t_hat) + self.epsilon)

            #print(theta)
            # termination condition
            if sum(abs(theta - theta_prev)) <= self.epsilon or epoch == self.iterations:
                break

        return theta

    def calculate_adam_accuracy(X_train, y_train, X_test, y_test):
        file = open("adam.txt", 'r')
        print("iter: test_word, test_letter, train_word, train_letter, train_word")
        for line in file:
            split = line.split()
            print(split[0] + ": ")
            params = np.array(split[1:]).astype(np.float)
            fg.print_accuracies(params, X_train, y_train, X_test, y_test)
        file.close()

X_train, y_train = gd.read_data("train_sgd.txt")
X_test, y_test = gd.read_data("test_sgd.txt")
params = np.zeros(129*26 + 26 **2)
cf = cf.callback_function(X_train, y_train,  X_test, y_test, "adam_1e-2.txt", 1e-2)
cf.delete_file()
print("computing optimal params")
#args are callbackfunction,      lambda,   learning rate, max iters, and gtol

adam = Adam(34000)
opt_params = adam.update(cf)

print("Final accuracy:")
fg.print_accuracies(opt_params, X_train, y_train, X_test, y_test)