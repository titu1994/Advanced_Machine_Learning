import numpy as np
from utils import save_params, save_losses, remove_file

"""
Gradient Computation
"""

# compute the forward pass (alpha)
def compute_forward_message(energies, T):
    word_len = len(energies)

    M = np.zeros((word_len, 26))
    for i in range(1, word_len):
        vect = M[i - 1] + T.transpose()
        vect_max = np.max(vect, axis=1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + energies[i - 1]), axis=-1))

    return M


# compute the backward pass (beta)
def compute_backward_message(energies, T):
    fin_index = len(energies) - 1

    #only need to go from the end to stated position
    M = np.zeros((len(energies), 26))
    t_trans = T

    for i in range(fin_index -1, -1, -1):
        vect = M[i + 1] + t_trans
        vect_max = np.max(vect, axis = 1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + energies[i + 1]), axis =  1))

    return M


# compute numerator of a single word
def compute_numerator(energies, y, T):
    total = 0
    for i in range(len(energies)):
        total += energies[i][y[i]]

        if(i > 0):
            total += T[y[i - 1]][y[i]]

    return np.exp(total)


# compute the denominator of a single word
def compute_denominator(alpha, energies):
    return np.sum(np.exp(alpha[-1] + energies[-1]))


def matricize_W(params):
    w = np.copy(params[:26 * 129])
    w = w.reshape((26, 129))
    return w


def matricize_Tij(params):
    t_ij = np.copy(params[26 * 129:])
    t_ij = t_ij.reshape((26, 26)).T
    return t_ij


def compute_gradient_wrt_Wy(X, y, energies, T, alpha, beta, denominator):
    gradient = np.zeros((26, 129))

    for i in range(len(X)):
        gradient[y[i]] += X[i]
        # for each position subtract off the probability of the letter
        temp = np.ones((26, 129)) * X[i]
        temp = temp.transpose() * np.exp(alpha[i] + beta[i] + energies[i]) / denominator
        gradient -= temp.transpose()

    return gradient.flatten()


def compute_gradient_wrt_Tij(energies, y, T, alpha, beta, denominator):
    gradient = np.zeros((26, 26))

    for i in range(len(energies) - 1):
        energy = energies[i]
        alph = alpha[i]

        for j in range(26):
            gradient[j, :] -= np.exp(energy + T.transpose()[j] + energies[i + 1][j] + beta[i + 1][j] + alph)

    gradient /= denominator
    gradient = gradient.flatten()

    for i in range(len(energies) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def compute_gradient_per_word(X, y, W, T, word_id):
    energies = np.inner(X[word_id], W)

    alpha = compute_forward_message(energies, T)
    beta = compute_backward_message(energies, T)
    denominator = compute_denominator(alpha, energies)

    W_grad = compute_gradient_wrt_Wy(X[word_id], y[word_id], energies, T, alpha, beta, denominator)
    T_grad = compute_gradient_wrt_Tij(energies, y[word_id], T, alpha, beta, denominator)

    return np.concatenate((W_grad, T_grad))


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
    f_mess = compute_forward_message(w_x, t)
    b_mess = compute_backward_message(w_x, t)
    den = compute_denominator(f_mess, w_x)
    wy_grad = grad_wrt_wy_mcmc(X[word_num], y[word_num], w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels)
    t_grad = grad_wrt_t_mcmc(y[word_num], w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels)

    return np.concatenate((wy_grad, t_grad))


def grad_wrt_wy_mcmc(X, y, w_x, t, f_mess, b_mess, den, sampled_indices, sampled_letters, sampled_labels):
    gradient = np.zeros((26, 129))
    for i in range(len(X)):
        gradient[y[i]] += X[i]
        # for each position subtract off the probability of the letter
        temp = np.ones((26, 129)) * X[i]
        temp = temp.transpose() * np.exp(f_mess[i] + b_mess[i] + w_x[i]) / den
        gradient -= temp.transpose()
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


def averaged_gradient(params, X, y, limit):
    W = matricize_W(params)
    T = matricize_Tij(params)

    total = np.zeros(129 * 26 + 26 ** 2)
    for i in range(limit):
        total += compute_gradient_per_word(X, y, W, T, i)

    return total / limit


def compute_log_p_y_given_x(energies, y, T, word_id):
    alpha = compute_forward_message(energies, T)
    factor = compute_numerator(energies, y, T) / compute_denominator(alpha, energies)
    return np.log(factor)


def compute_log_p_y_given_x_average(params, X, y, limit):
    W = matricize_W(params)
    T = matricize_Tij(params)

    total = 0
    for i in range(limit):
        w_x = np.inner(X[i], W)
        total += compute_log_p_y_given_x(w_x, y[i], T, i)

    return total / limit


"""
Optimization
"""

def objective_function(params, X_train, y_train, lambd):
    num_examples = len(X_train)
    l2_regularization = 1 / 2 * np.sum(params ** 2)
    mean_logloss = compute_log_p_y_given_x_average(params, X_train, y_train, num_examples)
    return -mean_logloss + lambd * l2_regularization


def objective_function_per_word(params, X_train, y_train, word_id, lambd):
    W = matricize_W(params)
    T = matricize_Tij(params)
    energies = np.inner(X_train[word_id], W)

    l2_regularization = 1 / 2 * np.sum(params ** 2)
    mean_logloss = compute_log_p_y_given_x(energies, y_train[word_id], T, word_id)

    return -mean_logloss + lambd * l2_regularization


def d_optimization_function(params, X_train, y_train, lambd):
    num_examples = len(X_train)

    grad_mean_logloss = averaged_gradient(params, X_train, y_train, num_examples)
    grad_l2_regularization = params

    return -grad_mean_logloss + lambd * grad_l2_regularization


def d_optimization_function_per_word(params, X_train, y_train, word_id, lambd):
    W = matricize_W(params)
    T = matricize_Tij(params)

    grad_mean_logloss = compute_gradient_per_word(X_train, y_train, W, T, word_id)
    grad_l2_regularization = params

    return -grad_mean_logloss + lambd * grad_l2_regularization


def grad_func_word_mcmc(params, X_train, y_train, word_num, l, num_samples):
    w = matricize_W(params)
    t = matricize_Tij(params)
    grad_avg = gradient_word_mcmc(X_train, y_train, w, t, word_num, num_samples)
    grad_reg = params
    return - grad_avg + l * grad_reg


def print_averaged_gradient(params, X_train, y_train, lambd):
    avg_grad = d_optimization_function(params, X_train, y_train, lambd)

    print("Average gradient: ", np.sum(avg_grad ** 2))
    print()


class Callback(object):
    '''
    Helper class used for printing and storing the parameters every epoch of optimization.
    Also used for printing and saving logloss.
    '''

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
        loss = objective_function(params, self.X, self.y, self.lambd)
        print(loss)

        print("Average gradient: ", end='')
        avg_grad = np.mean(d_optimization_function(params, self.X, self.y, self.lambd) ** 2)
        print(avg_grad)
        print()

        self.iters += 1
        save_params(params, self.filename, self.iters)
        save_losses(loss, self.loss_filename, self.iters)


    def callback_fn_return_vals(self, params):
        print("Function value: ", end='')
        loss = objective_function(params, self.X, self.y, self.lambd)
        print(loss)

        print("Average gradient: ", end='')
        avg_grad = np.mean(d_optimization_function(params, self.X, self.y, self.lambd) ** 2)
        print(avg_grad)
        print()

        self.iters += 1
        save_params(params, self.filename, self.iters)
        save_losses(loss, self.loss_filename, self.iters)
        return loss, avg_grad
