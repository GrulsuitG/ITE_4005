import sys
import time

import numpy as np
import pandas as pd


class Recommend:
    def __init__(self):
        self.mean = 2
        self.user = None
        self.item = None
        self.user_mat = None
        self.item_mat = None
        self.bias_user = None
        self.bias_item = None

        self.lr = 0.005
        self.reg = 0.02
        self.factors = 100
        self.epochs = 20


    def fit(self, X, y):
        unique_user = np.unique(X[:, 0])
        unique_item = np.unique(X[:, 1])

        bias_user = np.zeros(unique_user.size, np.double)
        bias_item = np.zeros(unique_item.size, np.double)

        user_mat = np.random.normal(0, 0.1, (len(unique_user), self.factors))
        item_mat = np.random.normal(0, 0.1, (self.factors, len(unique_item)))

        mean = np.mean(y)

        for _ in range(self.epochs):
            for (u, i), r in zip(X, y):
                u = np.where(unique_user == u)[0][0]
                i = np.where(unique_item == i)[0][0]
                predict_r = mean + bias_user[u] + bias_item[i] + np.dot(user_mat[u, :], item_mat[:, i])
                err = r - predict_r

                bias_user[u] += self.lr * (err - self.reg * bias_user[u])
                bias_item[i] += self.lr * (err - self.reg * bias_item[i])
                for k in range(self.factors):
                    user_mat[u, k] += self.lr * (err * item_mat[k, i] - self.reg * user_mat[u, k])
                    item_mat[k, i] += self.lr * (err * user_mat[u, k] - self.reg * item_mat[k, i])

        self.user = unique_user
        self.item = unique_item
        self.bias_user = bias_user
        self.bias_item = bias_item
        self.user_mat = user_mat
        self.item_mat = item_mat
        self.mean = mean

    def predict(self, test):
        return [
            self.predict_pair(u, i) for u, i in zip(test['user_id'], test['item_id'])
        ]

    def predict_pair(self, u, i):
        user_known, item_known = False, False
        pred = self.mean

        if u in self.user:
            user_known = True
            u = np.where(self.user == u)[0][0]
            pred += self.bias_user[u]

        if i in self.item:
            item_known = True
            i = np.where(self.item == i)[0][0]
            pred += self.bias_item[i]

        if user_known and item_known:
            pred += np.dot(self.user_mat[u, :], self.item_mat[:, i])

        pred = min(5, pred)
        pred = max(1, pred)

        return pred


def main():
    argv = sys.argv

    if len(argv) == 3:
        training_file = argv[1]
        test_file = argv[2]
    else:
        print("usage :", argv[0].split('/')[-1], "<training file> <test file>")
        exit(0)

    output_file = test_file.split('.')[0] + '.base_prediction.txt'

    training = pd.read_csv(training_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    test = pd.read_csv(test_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    training = training.drop(columns='timestamp')

    test = test.drop(columns='timestamp')

    rec = Recommend()
    rec.fit(training.values[:, :2], training.values[:, 2])

    test['rating'] = rec.predict(test)
    test.to_csv(output_file, sep='\t', index=False, header=None)


if __name__ == '__main__':
    main()
