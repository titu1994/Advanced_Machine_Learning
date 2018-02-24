from sklearn.datasets.svmlight_format import load_svmlight_file

Q1_TRAIN_PATH = "data/train.txt"
Q1_TEST_PATH = "data/test.txt"


def load_Q1_data():
    X_train, y_train = load_svmlight_file(Q1_TRAIN_PATH)
    X_test, y_test = load_svmlight_file(Q1_TEST_PATH)

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_Q1_data()

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)