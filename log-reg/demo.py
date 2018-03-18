import numpy as np
import random


class LogisticRegression:
    def __init__(self, learning_rate=1, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.__cost_history = []

    def fit(self, X, y, initial_weights=None):
        X = self.__fsmn(X)
        X = np.append(np.ones((len(X), 1)), X, axis=1)
        y = y.reshape((len(y), 1))
        if initial_weights is None:
            initial_weights = np.zeros(len(X[0]))

        self.weights = initial_weights.reshape((len(initial_weights), 1))
        cost = self.__get_cost(X, y)

        for i in range(self.max_iterations):
            self.__cost_history.append([i, cost])
            hypothesis = self.__hypothesis(X)
            new_weights = self.weights - \
                (np.mean(X * (hypothesis - y),
                         axis=0).reshape((len(self.weights), 1))) * self.learning_rate
            new_cost = self.__get_cost(X, y)
            if new_cost <= cost:
                cost = new_cost
                self.weights = new_weights
            else:
                break

    def predict(self, X):
        X = self.__fsmn(X)
        X = np.append(np.ones((len(X), 1)), X, axis=1)
        hypothesis = self.__hypothesis(X)
        hypothesis[hypothesis >= 0.5] = 1
        hypothesis[hypothesis < 0.5] = 0
        return hypothesis.reshape((1, len(hypothesis))).tolist()[0]

    def score(self, actual_output, expected_output):
        return np.sum(expected_output == actual_output) / len(actual_output)

    def plot_cost_history(self):
        import matplotlib.pyplot as plt
        self.__cost_history = np.array(self.__cost_history)
        plt.plot(self.__cost_history[:, 0], self.__cost_history[:, 1])
        plt.xlabel('Iteration Number')
        plt.ylabel('Cost')
        plt.show()

    def __hypothesis(self, X):
        return 1/(1+np.exp(-np.matmul(X, self.weights)))

    def __get_cost(self, X, y):
        hypothesis = self.__hypothesis(X)
        return np.mean(-y * np.log(hypothesis) - (1-y) * np.log(1-hypothesis))

    def __fsmn(self, data):
        return (data - np.mean(data, axis=0)) / np.ptp(data, axis=0)


def main():
    data = np.genfromtxt('train.csv', delimiter=',')
    data = data[1:]
    train_len = int(len(data) * 0.70)
    train_data = data[:train_len]
    test_data = data[train_len:]
    clf = LogisticRegression()
    clf.fit(train_data[:, :-1], train_data[:, -1])
    y = clf.predict(test_data[:, :-1])
    print("Accuracy:", clf.score(y, test_data[:, -1]))
    clf.plot_cost_history()


if __name__ == '__main__':
    main()
