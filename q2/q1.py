import numpy as np
from copy import deepcopy


def get_energies(X, w, t):
    # for 100 letters I need an M matrix of 100  X 26
    M = np.zeros((len(X), 26))

    # populates first row
    for j in range(26):
        M[0][j] = np.inner(X[0], w[j])

    # go row wise through M matrix, starting at line 2 since first line is populated
    for row in range(1, len(X)):

        # go column wise, populating the best sum of the previous + T[previous letter][
        for cur_letter in range(26):
            # initialize with giant negative number
            best = -99999999999999

            # iterate over all values of the previous letter, fixing the current letter
            for prev_letter in range(26):
                temp_product = M[row - 1][prev_letter] + np.inner(X[row], w[cur_letter]) + t[prev_letter][cur_letter]
                if (temp_product > best):
                    best = temp_product
            M[row][cur_letter] = best
    return M


def trial_solution_value(trial_solution, X, w, t):
    running_total = np.inner(X[0], w[trial_solution[0]])
    for i in range(1, len(X)):
        running_total += np.inner(X[i], w[trial_solution[i]]) + t[trial_solution[i - 1]][trial_solution[i]]
    return running_total


def optimize_brute_force(X, w, t):
    stopping_criteria = []
    trial_solution = []
    # initialize stopping criteria and trial solutions
    for i in range(len(X)):
        stopping_criteria.append(25)
        trial_solution.append(0)

        # increment trial solution
    best = trial_solution_value(trial_solution, X, w, t)
    best_solution = trial_solution
    while (True):
        i = 0
        while (True):
            trial_solution[i] += 1
            if (trial_solution[i] == 26):
                trial_solution[i] = 0
                i += 1
            else:
                break

        trial_value = trial_solution_value(trial_solution, X, w, t)
        if (trial_value > best):
            best = trial_value
            best_solution = deepcopy(trial_solution)

        if (trial_solution == stopping_criteria):
            break
    return best, best_solution


def predict(X, w, t):
    print("Predicting %d characters" % len(X))

    M = get_energies(X, w, t)
    solution = []
    cur_word_pos = len(M) - 1
    prev_word_pos = cur_word_pos - 1
    cur_letter = np.argmax(M[cur_word_pos])
    cur_val = M[cur_word_pos][cur_letter]
    solution.insert(0, cur_letter)

    while (cur_word_pos > 0):
        for prev_letter in range(26):
            if (abs(cur_val - M[prev_word_pos][prev_letter] - t[prev_letter][cur_letter] - np.inner(X[cur_word_pos], w[cur_letter])) < 0.00001):
                solution.insert(0, prev_letter)
                cur_letter = prev_letter
                cur_word_pos -= 1
                prev_word_pos -= 1
                cur_val = M[cur_word_pos][cur_letter]
                break

    return np.array(solution)


def get_weights():
    file = open('data/decode_input.txt', 'r')
    x_array = []
    w_array = []
    t_array = []
    for i, elt in enumerate(file):
        if (i < 100 * 128):
            x_array.append(elt)
        elif (i < 100 * 128 + 128 * 26):
            w_array.append(elt)
        else:
            t_array.append(elt)
    return x_array, w_array, t_array


def parse_x(x):
    count = 0
    x_array = np.zeros((100, 128))
    for i in range(100):
        for j in range(128):
            x_array[i][j] = x[count]
            count += 1
    return x_array


def parse_w(w):
    w_array = np.zeros((26, 128))
    count = 0
    for i in range(26):
        for j in range(128):
            w_array[i][j] = w[count]
            count += 1
    return w_array


def parse_t(t):
    t_array = np.zeros((26, 26))
    count = 0
    for i in range(26):
        for j in range(26):
            # this is actuqlly right.  it goes T11, T21, T31...
            t_array[j][i] = t[count]
            count += 1
    return t_array


def get_weights_formatted():
    x, w, t = get_weights()
    x_array = parse_x(x)
    w_array = parse_w(w)
    t_array = parse_t(t)
    return x_array, w_array, t_array


if __name__ == '__main__':
    import numpy as np

    X, w, t = get_weights_formatted()
    soln = predict(X, w, t)

    with open("result/decode_output.txt", "w") as text_file:
        for i, elt in enumerate(soln):
            elt = str(chr(elt + ord('A')))
            text_file.write(str(elt))
            text_file.write("\n")
