import numpy as np
from utils import save_params, save_losses, remove_file

"""
Gradient Computation
"""

def forward_propogate(w_x, t):
    word_len = len(w_x)
    # establish matrix to hold results
    M = np.zeros((word_len, 26))

    # iterate through length of word
    for i in range(1, word_len):
        vect = M[i - 1] + t.transpose()
        vect_max = np.max(vect, axis=1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + w_x[i - 1]), axis=1))

    return M


def back_propogate(w_x, t):
    #get the index of the final letter of the word
    fin_index = len(w_x) - 1

    #only need to go from the end to stated position
    M = np.zeros((len(w_x), 26))
    #now we need taa, tab, tac... because we are starting at the end and working backwards
    #which is exactly the transposition of the t matrix
    t_trans = t

    for i in range(fin_index -1, -1, -1):
        vect = M[i + 1] + t_trans
        vect_max = np.max(vect, axis = 1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max +np.log(np.sum(np.exp(vect + w_x[i+1]), axis =  1))
    return M


def numerator(w_x, y, t):
    #full numerator for an entire word
    total = 0
    #go through whole word
    for i in range(len(w_x)):
        #no matter what add W[actual letter] inner Xi
        total += w_x[i][y[i]]
        if(i > 0):
            #again we have t stored as Tcur, prev
            total += t[y[i-1]][y[i]]
    return np.exp(total)


def denominator(f_message, w_x):
    #this is  eassy, just forward propogate to the end of the word and return the sum of the exponentials
    return np.sum(np.exp(f_message[-1] + w_x[-1]))


#split up params into w and t.  Note that this only needs to happen once per word!!! do not calculate per letter
def matricize_W(params):
    w = np.copy(params[:26 * 129])
    w = w.reshape((26, 129))
    return w

def matricize_Tij(params):
    t_ij = np.copy(params[26 * 129:])
    t_ij = t_ij.reshape((26, 26)).T
    return t_ij


def grad_wrt_wy(X, y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros((26, 129))
    for i in range(len(X)):
        gradient[y[i]] += X[i]
        # for each position subtract off the probability of the letter
        temp = np.ones((26, 129)) * X[i]
        temp = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
        gradient -= temp.transpose()
    return gradient.flatten()


def grad_wrt_t(y, w_x, t, f_mess, b_mess, den):
    gradient = np.zeros(26 * 26)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(
                w_x[i] + t.transpose()[j] + w_x[i + 1][j] + b_mess[i + 1][j] + f_mess[i])

    gradient /= den

    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def gradient_word(X, y, w, t, word_num):
    w_x = np.inner(X[word_num], w)
    f_mess = forward_propogate(w_x, t)
    b_mess = back_propogate(w_x, t)
    den = denominator(f_mess, w_x)
    wy_grad = grad_wrt_wy(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den)
    t_grad = grad_wrt_t(y[word_num], w_x, t, f_mess, b_mess, den)
    return np.concatenate((wy_grad, t_grad))


def gradient_word_mcmc(X, y, w, t, word_num, num_samples):
    sampled_letters = []
    sampled_labels = []
    sampled_indices = []
    for i in range(num_samples):
        indices = np.arange(len(X[word_num]), dtype=int)
        index = int(np.random.choice(indices, size=1))

        sampled_indices.append(index)
        sampled_letter = X[word_num][index]
        sampled_label = y[word_num][index]

        sampled_letters.append(sampled_letter)
        sampled_labels.append(sampled_label)

    w_x = np.inner(X[word_num], w)
    f_mess = forward_propogate(w_x, t)
    b_mess = back_propogate(w_x, t)
    den = denominator(f_mess, w_x)
    wy_grad = grad_wrt_wy_mcmc(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels)
    t_grad = grad_wrt_t_mcmc(y[word_num], w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels)

    return np.concatenate((wy_grad, t_grad))


def grad_wrt_wy_mcmc(X, y, w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels):
    gradient = np.zeros((26, 129))

    for i in range(len(sampled_letters)):
        prior_dist = np.ones((26, 129)) * sampled_letters[i]
        print(prior_dist)
    # gradient = np.zeros((26, 129))
    # for i in range(len(X)):
    #     gradient[y[i]] += X[i]
    #     # for each position subtract off the probability of the letter
    #     temp = np.ones((26, 129)) * X[i]
    #     temp = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
    #     gradient -= temp.transpose()
    # return gradient.flatten()

    # for i in range(len(sampled_indices)):
    #     temp = np.ones((26, 129)) * sampled_letters[i]
    #     for j in range(26):
    #         gradient[j] += sampled_letters[i]
    #         # temp = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
    #         t = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
    #         print(t.shape)
    #         print(temp.shape)
    #         temp *= t.transpose()
    #         # temp = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
    #         # fix this
    #     gradient -= temp

    return gradient.flatten()


def grad_wrt_t_mcmc(y, w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels):
    gradient = np.zeros(26 * 26)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(
                w_x[i] + t.transpose()[j] + w_x[i + 1][j] + b_mess[i + 1][j] + f_mess[i])

    gradient /= den

    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def gradient_avg(params, X, y, up_to_index):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = np.zeros(129 * 26 + 26 ** 2)
    for i in range(up_to_index):
        total += gradient_word(X, y, w, t, i)
    return total / (up_to_index)


def log_p_y_given_x(w_x, y, t, word_num):
    f_mess = forward_propogate(w_x, t)
    return np.log(numerator(w_x, y, t) / denominator(f_mess, w_x))


def log_p_y_given_x_avg(params, X, y, up_to_index):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = 0
    for i in range(up_to_index):
        w_x = np.inner(X[i], w)
        total += log_p_y_given_x(w_x, y[i], t, i)
    return total / (up_to_index)


"""
Optimization
"""

def func_to_minimize(params, X_train, y_train, l):
    num_examples = len(X_train)
    reg = 1 / 2 * np.sum(params ** 2)
    avg_prob = log_p_y_given_x_avg(params, X_train, y_train, num_examples)
    return -avg_prob + l * reg


def func_to_minimize_word(params, X_train, y_train, word_num, l):
    w = matricize_W(params)
    t = matricize_Tij(params)
    reg = 1 / 2 * np.sum(params ** 2)
    w_x = np.inner(X_train[word_num], w)
    prob = log_p_y_given_x(w_x, y_train[word_num], t, word_num)

    return -prob + l * reg


def grad_func(params, X_train, y_train, l):
    num_examples = len(X_train)
    grad_avg = gradient_avg(params, X_train, y_train, num_examples)
    grad_reg = params
    return - grad_avg + l * grad_reg


def grad_func_word(params, X_train, y_train, word_num, l):
    w = matricize_W(params)
    t = matricize_Tij(params)
    grad_avg = gradient_word(X_train, y_train, w, t, word_num)
    grad_reg = params
    return - grad_avg + l * grad_reg


def grad_func_word_mcmc(params, X_train, y_train, word_num, l, num_samples):
    w = matricize_W(params)
    t = matricize_Tij(params)
    grad_avg = gradient_word_mcmc(X_train, y_train, w, t, word_num, num_samples)
    grad_reg = params
    return - grad_avg + l * grad_reg


def print_gradient_average(params, X_train, y_train, lambd):
    avg_grad = grad_func(params, X_train, y_train, lambd)

    print("Average gradient: ", np.sum(avg_grad ** 2))
    print()


class Callback(object):

    def __init__(self, X_train, y_train, filename, lambd):
        self.X = X_train
        self.y = y_train

        self.filename = "results/" + filename
        self.loss_filename = "results/" + filename[:-4] + "_f_evals.txt"
        self.lambd = lambd
        self.iters = 0

        remove_file(self.filename)
        remove_file(self.loss_filename)

    def callback_fn(self, params):
        print("Function value: ", end='')
        loss = func_to_minimize(params, self.X, self.y, self.lambd)
        print(loss)

        print("Average gradient: ", end='')
        avg_grad = np.mean(grad_func(params, self.X, self.y, self.lambd) ** 2)
        print(avg_grad)
        print()

        self.iters += 1
        save_params(params, self.filename, self.iters)
        save_losses(loss, self.loss_filename, self.iters)


    def callback_fn_return_vals(self, params):
        print("Function value: ", end='')
        loss = func_to_minimize(params, self.X, self.y, self.lambd)
        print(loss)

        print("Average gradient: ", end='')
        avg_grad = np.mean(grad_func(params, self.X, self.y, self.lambd) ** 2)
        print(avg_grad)
        print()

        self.iters += 1
        save_params(params, self.filename, self.iters)
        save_losses(loss, self.loss_filename, self.iters)
        return loss, avg_grad
