import numpy as np
import time
from scipy.optimize import check_grad, fmin_bfgs

from project2.utils import prepare_structured_dataset, load_model_params, load_model_params_zeros, compute_word_char_accuracy_score
from project2.crf_evaluate import decode_crf


def compute_forward_message(w_x, t):
    word_len = len(w_x)
    M = np.zeros((word_len, 26))
    # set first row to inner <wa, x0> <wb, x0>...

    # iterate through length of word
    for i in range(1, word_len):
        vect = M[i - 1] + t.transpose()
        vect_max = np.max(vect, axis=1)
        vect = (vect.transpose() - vect_max).transpose()
        M[i] = vect_max + np.log(np.sum(np.exp(vect + w_x[i - 1]), axis=1))

    return M

def compute_backward_message(w_x, t):
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
        M[i] = vect_max +np.log(np.sum(np.exp(vect + w_x[i+1]), axis =1))

    return M


def compute_numerator(w_x, y, t):
    # full compute_numerator for an entire word
    total = 0
    # go through whole word
    for i in range(len(w_x)):
        # no matter what add W[actual letter] inner Xi
        total += w_x[i][y[i]]
        if (i > 0):
            # again we have t stored as Tcur, prev
            total += t[y[i - 1]][y[i]]
    return np.exp(total)


def compute_denominator(w_x, alpha):
    #return np.sum(np.exp(alpha[-1] + w_x[-1]))

    # forward propagate to the end of the word and return the sum
    intermediate = alpha[-1] + w_x[-1]
    max_inter = np.max(intermediate)
    intermediate -= max_inter
    out = max_inter + np.log(np.sum(np.exp(intermediate)))
    return out

# convert W into a matrix from its flattened parameters
def matricize_W(params):
    w = params[:26 * 129]
    w = w.reshape((26, 129))
    return w


def matricize_Tij(params):
    t_ij = params[26 * 129:]
    t_ij = t_ij.reshape((26, 26)).T
    return t_ij


def compute_gradient_wrt_Wy(X, y, w_x, alpha, beta, denominator):
    gradient = np.zeros((26, 129))

    for i in range(len(X)):
        gradient[y[i]] += X[i]
        # for each position subtract off the probability of the letter
        temp = np.ones((26, 129)) * X[i]
        temp = temp.transpose() * np.exp(alpha[i] + beta[i] + w_x[i] - denominator)
        gradient -= temp.transpose()

    return gradient.flatten()


def compute_gradient_wrt_Tij(w_x, y, t, alpha, beta, denominator):
    gradient = np.zeros(26 * 26)
    for i in range(len(w_x) - 1):
        for j in range(26):
            gradient[j * 26: (j + 1) * 26] -= np.exp(w_x[i] + t.transpose()[j] + w_x[i + 1][j] + beta[i + 1][j] + alpha[i] - denominator)

    #gradient /= compute_denominator

    for i in range(len(w_x) - 1):
        t_index = y[i]
        t_index += 26 * y[i + 1]
        gradient[t_index] += 1

    return gradient


def gradient_per_word(X, y, w, t, word_index, concat_grads=True):
    w_x = np.inner(X[word_index], w)

    alpha = compute_forward_message(w_x, t)
    beta = compute_backward_message(w_x, t)
    denominator = compute_denominator(w_x, alpha)
    wy_grad = compute_gradient_wrt_Wy(X[word_index], y[word_index], w_x, alpha, beta, denominator)
    t_grad = compute_gradient_wrt_Tij(w_x, y[word_index], t, alpha, beta, denominator)

    if concat_grads:
        return np.concatenate((wy_grad, t_grad))
    else:
        wy_grad = wy_grad.reshape((26, 129))
        t_grad = t_grad.reshape((26, 26))
        return wy_grad, t_grad


def averaged_gradient(params, X, y, limit):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = np.zeros(129 * 26 + 26 ** 2)
    for i in range(limit):
        total += gradient_per_word(X, y, w, t, i)
    return total / (limit)


def compute_log_p_y_given_x(w_x, y, t, word_index):
    f_mess = compute_forward_message(w_x, t)
    return np.log(compute_numerator(w_x, y, t) / np.exp(compute_denominator(w_x, f_mess)))


def compute_log_p_y_given_x_average(params, X, y, limit):
    w = matricize_W(params)
    t = matricize_Tij(params)

    total = 0
    for i in range(limit):
        w_x = np.inner(X[i], w)
        total += compute_log_p_y_given_x(w_x, y[i], t, i)
    return total / (limit)


def optimization_function(params, X, y, C, lambd, limit):
    num_examples = len(X)
    reg = 1/2 * np.sum(params ** 2)
    avg_prob = compute_log_p_y_given_x_average(params, X, y, limit)
    return -C * avg_prob + lambd * reg


def d_optimization_function(params, X, y, C, lambd, limit):
    logloss_gradient = averaged_gradient(params, X, y, limit)
    l2_loss_gradient = lambd * params
    return -C * logloss_gradient + l2_loss_gradient


def optimization_function_word(X, y, w, t, word_index, C, lambd):
    l2_loss = 1 / 2 * np.sum(params ** 2)

    w_x = np.inner(X[word_index], w)
    logloss = compute_log_p_y_given_x(w_x, y, t, word_index)

    return -C * logloss + lambd * l2_loss


def d_optimization_function_word(X, y, w, t, word_index, C, lambd):
    dW, dT = gradient_per_word(X, y, w, t, word_index, concat_grads=False)
    dW = -C * dW + lambd * w
    dT = -C * dT + lambd * t
    return [dW, dT]


def check_gradient(params, X, y):
    # check the gradient of the first 10 words
    grad_value = check_grad(compute_log_p_y_given_x_average, averaged_gradient, params, X, y, 1)
    print("Gradient check (first word) : ", grad_value)


def check_gradient_optimization(params, X, y):
    # check the gradient of the first 10 words
    grad_value = check_grad(optimization_function, d_optimization_function, params, X, y, 1, 1., 1)
    print("Gradient optimization check (first word) : ", grad_value)


def train_crf_lbfgs(params, X, y, X_test, y_test, C, lambd, model_name):
    start = time.time()
    out = fmin_bfgs(optimization_function, params, d_optimization_function, (X, y, C, lambd), gtol = 0.01)
    print("Total time: ", end = '')
    print(time.time() - start)

    with open("result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt) + "\n")



def train_crf_sgd(params, X, y, C, num_epochs, learning_rate, l2_lambda, test_xy, model_name):
    num_words = len(X)
    test_X, test_Y = test_xy

    W = matricize_W(params)
    T = matricize_Tij(params)

    print("Starting SGD optimizaion")

    # make results reproducible
    np.random.seed(1)

    W_running_avg = np.zeros_like(W)
    T_running_avg = np.zeros_like(T)

    MOMENTUM = 0.9

    W_old_avg = np.copy(W_running_avg)
    T_old_avg = np.copy(T_running_avg)

    start = time.time()
    for epoch in range(num_epochs):
        print("Begin epoch %d" % (epoch + 1))

        indices = np.arange(0, num_words, step=1, dtype=int)
        np.random.shuffle(indices)

        for i, word_index in enumerate(indices):
            W_grad, T_grad = d_optimization_function_word(X, y, W, T, word_index, C, l2_lambda)

            W_running_avg = MOMENTUM * W_running_avg + (1. - MOMENTUM) * W_grad
            T_running_avg = MOMENTUM * T_running_avg + (1. - MOMENTUM) * T_grad

            # perform SGD update
            W -= learning_rate * W_running_avg
            T -= learning_rate * T_running_avg

            if i % 1000 == 0:
                params = np.concatenate((W.flatten(), T.flatten()))
                logloss = optimization_function(params, X, y, C, l2_lambda, num_words)
                print("Epoch %d Iter %d : Logloss : " % (epoch + 1, i), logloss)
                print("W norm", np.linalg.norm(W))
                print("T norm", np.linalg.norm(T))

            learning_rate *= 0.999
            learning_rate = max(learning_rate, 5e-4)

            #print("W norm", np.linalg.norm(W))
            #print("T norm", np.linalg.norm(T))
            #print("lr", learning_rate)
            #print("W grad stats", np.linalg.norm(W_grad), np.min(W_grad), np.max(W_grad), np.mean(W_grad), np.std(W_grad))
            #print("T grad stats", np.linalg.norm(T_grad), np.min(T_grad), np.max(T_grad), np.mean(T_grad), np.std(T_grad))
            #print()

        print("Learning rate : ", learning_rate)

        params = np.concatenate((W.flatten(), T.flatten()))
        logloss = optimization_function(params, X, y, C, l2_lambda, num_words)
        print("Logloss : ", logloss)
        print()

        moving_sum = np.sum(W_running_avg ** 2) + np.sum(T_running_avg ** 2)
        old_moving_sum = np.sum(W_old_avg ** 2) + np.sum(T_old_avg ** 2)

        if abs(moving_sum - old_moving_sum) < 1e-3:
            print("Gradient difference is small enough. Early stopping..")
            break
        else:
            W_old_avg = W_running_avg
            T_old_avg = T_running_avg

        # if (epoch + 1) % 5 == 0:
        #     print('*' * 80)
        #     print("Computing metrics after end of epoch %d" % (epoch + 1))
        #     # print evaluation metrics every 1000 steps of SGD
        #     train_loss = optimization_function(params, X, y, C, l2_lambda, num_words)
        #
        #     y_preds = decode_crf(test_X, W, T)
        #     word_acc, char_acc = compute_word_char_accuracy_score(y_preds, test_Y)
        #
        #     print("Epoch %d | Train loss = %0.8f | Word Accuracy = %0.5f | Char Accuracy = %0.5f" %
        #           (epoch + 1, train_loss, word_acc, char_acc))
        #     print('*' * 80, '\n')

    print('*' * 80)
    print("Computing metrics after end of epoch")
    # print evaluation metrics every 1000 steps of SGD
    train_loss = optimization_function(params, X, y, C, l2_lambda, num_words)

    y_preds = decode_crf(test_X, W, T)
    word_acc, char_acc = compute_word_char_accuracy_score(y_preds, test_Y)

    print("Epoch %d | Train loss = %0.8f | Word Accuracy = %0.5f | Char Accuracy = %0.5f" %
          (epoch + 1, train_loss, word_acc, char_acc))
    print('*' * 80, '\n')

    # merge the two grads into a single long vector
    out = np.concatenate((W.flatten(), T.flatten()))
    print("Total time: ", end='')
    print(time.time() - start)

    with open("result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt) + "\n")


def train_crf_adam(params, X, y, C, num_epochs, learning_rate, l2_lambda, test_xy, model_name,
                   beta1=0.9, beta2=0.999, epsilon=1e-8, amsgrad=False):
    num_words = len(X)
    test_X, test_Y = test_xy

    W = matricize_W(params)
    T = matricize_Tij(params)

    print("Starting SGD optimizaion")

    # make results reproducible
    np.random.seed(1)

    # ADAM parameters
    M = {'W': np.zeros_like(W, dtype=np.float32), 'T': np.zeros_like(T, dtype=np.float32)}
    R = {'W': np.zeros_like(W, dtype=np.float32), 'T': np.zeros_like(T, dtype=np.float32)}

    if amsgrad:
        R_hat = {'W': np.zeros_like(W, dtype=np.float32), 'T': np.zeros_like(T, dtype=np.float32)}

    start = time.time()
    iter = 1
    for epoch in range(num_epochs):
        print("Begin epoch %d" % (epoch + 1))

        indices = np.arange(0, num_words, step=1, dtype=int)
        np.random.shuffle(indices)

        for i, word_index in enumerate(indices):
            W_grad, T_grad = d_optimization_function_word(X, y, W, T, word_index, C, l2_lambda)

            # ADAM updates to Momentum params
            M['W'] = beta1 * M['W'] + (1 - beta1) * W_grad
            M['T'] = beta1 * M['T'] + (1 - beta1) * T_grad

            # ADAM updates to RMSProp params
            R['W'] = beta2 * R['W'] + (1 - beta2) * (W_grad ** 2)
            R['T'] = beta2 * R['T'] + (1 - beta2) * (T_grad ** 2)

            if not amsgrad:
                # ADAM Update
                # bias correction
                m_k_w = M['W'] / (1 - beta1 ** iter)
                m_k_t = M['T'] / (1 - beta1 ** iter)
                r_k_w = R['W'] / (1 - beta2 ** iter)
                r_k_t = R['T'] / (1 - beta2 ** iter)

                lr_m = learning_rate / (np.sqrt(r_k_w) + epsilon)
                lr_t = learning_rate / (np.sqrt(r_k_t) + epsilon)

                # perform ADAM update
                W -= lr_m * (m_k_w)
                T -= lr_t * (m_k_t)

            else:
                # AMSGrad Update
                R_hat['W'] = np.maximum(R_hat['W'], R['W'])
                R_hat['T'] = np.maximum(R_hat['T'], R['T'])

                # bias correction
                m_k_w = M['W'] / (1 - beta1 ** iter)
                m_k_t = M['T'] / (1 - beta1 ** iter)
                r_k_w = R_hat['W'] / (1 - beta2 ** iter)
                r_k_t = R_hat['T'] / (1 - beta2 ** iter)

                lr_m = learning_rate / (np.sqrt(r_k_w) + epsilon)
                lr_t = learning_rate / (np.sqrt(r_k_t) + epsilon)

                # perform AMSGrad update
                W -= lr_m * (m_k_w)
                T -= lr_t * (m_k_t)

                R['W'] = R_hat['W']
                R['T'] = R_hat['T']

            #learning_rate *= 0.99
            #learning_rate = max(learning_rate, 1e-5)
            #
            #print("W norm", np.linalg.norm(W))
            #print("T norm", np.linalg.norm(T))
            #print("lr", learning_rate)
            #print("W grad stats", np.linalg.norm(W_grad), np.min(W_grad), np.max(W_grad), np.mean(W_grad), np.std(W_grad))
            #print("T grad stats", np.linalg.norm(T_grad), np.min(T_grad), np.max(T_grad), np.mean(T_grad), np.std(T_grad))
            #print()

            iter += 1
            iter = min(iter, int(1e6))

        params = np.concatenate((W.flatten(), T.flatten()))
        print("W norm", np.linalg.norm(W))
        print("T norm", np.linalg.norm(T))
        logloss = optimization_function(params, X, y, C, num_words, num_words)
        print("Logloss : ", logloss)

        if (epoch + 1) % 5 == 0:
            print('*' * 80)
            print("Computing metrics after end of epoch %d" % (epoch + 1))
            # print evaluation metrics every 1000 steps of SGD
            train_loss = optimization_function(params, X, y, C, num_words, num_words)

            y_preds = decode_crf(test_X, W, T)
            word_acc, char_acc = compute_word_char_accuracy_score(y_preds, test_Y)

            print("Epoch %d | Train loss = %0.8f | Word Accuracy = %0.5f | Char Accuracy = %0.5f" %
                  (epoch + 1, train_loss, word_acc, char_acc))
            print('*' * 80, '\n')

    # merge the two grads into a single long vector
    out = np.concatenate((W.flatten(), T.flatten()))
    print("Total time: ", end='')
    print(time.time() - start)

    with open("result/" + model_name + ".txt", "w") as text_file:
        for i, elt in enumerate(out):
            text_file.write(str(elt) + "\n")


def get_trained_model_parameters(model_name):
    file = open('result/' + model_name + '.txt', 'r')
    params = []
    for i, elt in enumerate(file):
        params.append(float(elt))
    return np.array(params)


def get_trained_model_parameters_gradient_descent(model_name):
    file = open('result/' + model_name + '.txt', 'r')
    params = []
    for i, elt in enumerate(file.read().split()):
        params.append(float(elt))
    return np.array(params)



if __name__ == '__main__':

    ''' check gradients and write to file '''
    print("Checking gradient correctness")
    X, y = prepare_structured_dataset('train_sgd.txt')
    params = load_model_params()
    check_gradient(params, X, y)
    check_gradient_optimization(params, X, y)
    # measure_gradient_computation_time(params, X, y)

    w = matricize_W(params)
    t = matricize_Tij(params)

    word_index = 0
    lambd = 1e-2
    #optimization_val = optimization_function_word(X, y, w, t, word_index, C=1, lambd=lambd)
    W_grad, T_grad = d_optimization_function_word(X, y, w, t, word_index, C=1, lambd=lambd)

    print('norm(W grad)', np.linalg.norm(W_grad), 'W grad', W_grad)
    print('norm (T)', np.linalg.norm(T_grad), 'T grad', T_grad)

    exit()

    # sum_of_gradients = summed_gradient(params, X, y, len(X))
    # sum_grad_norm = np.linalg.norm(sum_of_gradients)
    # print('Stats (Sum of gradients) :', np.max(sum_of_gradients), np.min(sum_of_gradients), np.mean(sum_of_gradients), np.std(sum_of_gradients))
    # print("Norm Sum of grads", sum_grad_norm)

    # train_crf_lbfgs(params, X, y, C=1, model_name='lbfgs')

    # exit()

    ''' training  '''
    X_train, y_train = prepare_structured_dataset('train_sgd.txt')
    X_test, y_test = prepare_structured_dataset('test_sgd.txt')
    params = load_model_params_zeros()

    test_xy = [X_test, y_test]

    '''
    Run optimization. For C= 1000 it takes about an 56 minutes
    '''
    # Gradient based training (SGD) parameters
    # NUM_EPOCHS = 100
    # LEARNING_RATE = 0.005
    # L2_LAMBDA = 1e-2
    #
    # train_crf_sgd(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='sgd-2')

    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.005
    # L2_LAMBDA = 1e-4

    # train_crf_sgd(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='sgd-4')
    #
    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.005
    # L2_LAMBDA = 1e-6
    #
    # train_crf_sgd(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='sgd-6')

    # Gradient based training (ADAM + Optional AMSGrad) parameters
    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.001
    # L2_LAMBDA = 1e-2 # 1e-2
    #
    # AMSGRAD = True
    # BETA2 = 0.999
    #
    # train_crf_adam(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #                num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #                test_xy=test_xy, model_name='adam-2',
    #                beta2=BETA2, amsgrad=AMSGRAD)

    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.001
    # L2_LAMBDA = 1e-4
    #
    # AMSGRAD = True
    # BETA2 = 0.999
    #
    # train_crf_adam(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='adam-4',
    #                beta2=BETA2, amsgrad=AMSGRAD)
    #
    # NUM_EPOCHS = 1000
    # LEARNING_RATE = 0.001
    # L2_LAMBDA = 1e-6
    #
    # AMSGRAD = True
    # BETA2 = 0.999
    #
    # train_crf_adam(params, X_train, y_train, C=1, l2_lambda=L2_LAMBDA,
    #               num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
    #               test_xy=test_xy, model_name='adam-6',
    #                beta2=BETA2, amsgrad=AMSGRAD)

    params = get_trained_model_parameters_gradient_descent('sgd-2')
    w = matricize_W(params)
    t = matricize_Tij(params)

    print("Function value: ", optimization_function(params, X_train, y_train, C=1, lambd=L2_LAMBDA, limit=len(X_train)))

    ''' accuracy '''
    y_preds = decode_crf(X_test, w, t)

    with open("result/prediction-sgd.txt", "w") as text_file:
        for i, elt in enumerate(y_preds):
            # convert to characters
            for word in elt:
                text_file.write(str(word + 1))
                text_file.write("\n")

    print("Test accuracy : ", compute_word_char_accuracy_score(y_preds, y_test))
