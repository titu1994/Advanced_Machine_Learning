import numpy as np


def get_energies(X, w, t):
    # for 100 letters I need an M matrix of 100  X 26
    M = np.zeros((len(X), 26))

    # populates first row
    M[0] = np.inner(X[0], w)

    # go row wise through M matrix, starting at line 2 since first line is populated
    for row in range(1, len(X)):

        # go column wise, populating the best sum of the previous + T[previous letter][
        for cur_letter in range(26):
            # initialize with giant negative number
            best = -np.inf

            # iterate over all values of the previous letter, fixing the current letter
            for prev_letter in range(26):
                temp_product = M[row - 1][prev_letter] + np.inner(X[row], w[cur_letter]) + t[prev_letter][cur_letter]
                if (temp_product > best):
                    best = temp_product
            M[row][cur_letter] = best
    return M


def decode_crf_word(X, w, t):
    M = get_energies(X, w, t)

    cur_word_pos = len(M) - 1
    prev_word_pos = cur_word_pos - 1

    cur_letter = np.argmax(M[cur_word_pos])
    cur_val = M[cur_word_pos][cur_letter]

    solution = [cur_letter]

    while (cur_word_pos > 0):
        for prev_letter in range(26):
            energy = np.inner(X[cur_word_pos], w[cur_letter])
            if (np.isclose(cur_val - M[prev_word_pos][prev_letter] - t[prev_letter][cur_letter] - energy, 0,
                           rtol=1e-5)):
                solution.append(prev_letter)
                cur_letter = prev_letter
                cur_word_pos -= 1
                prev_word_pos -= 1
                cur_val = M[cur_word_pos][cur_letter]
                break

    solution = solution[::-1]  # reverse the prediction string
    return np.array(solution)


def decode_crf(X, w, t):
    y_pred = []
    for i, x in enumerate(X):
        preds = decode_crf_word(x, w, t)
        y_pred.append(preds)
    return y_pred
