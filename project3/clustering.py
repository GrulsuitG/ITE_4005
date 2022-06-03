import sys
from collections import defaultdict
import numpy as np
import pandas as pd

UNCLASSFIED = 999
OUTLIER = -1


class DBSCAN:
    def __init__(self, data, eps, minpts):
        self.data = data
        self.label = [UNCLASSFIED] * len(data)

        self.eps = eps
        self.minpts = minpts

        self.cluster_num = 0
        self.size = len(data)

        self.cluster = defaultdict(list)

    def fit(self):
        for idx in range(self.size):
            if self.label[idx] == UNCLASSFIED:
                if self.expand_cluster(idx):
                    self.cluster_num += 1

        for idx, value in enumerate(self.label):
            self.cluster[value].append(idx)

        self.cluster.pop(OUTLIER)

    def expand_cluster(self, idx):
        neighbors = self.get_neighbor(idx)
        if len(neighbors) < self.minpts:
            self.label[idx] = OUTLIER
            return False
        else:
            for neighbor in map(int, neighbors):
                self.label[neighbor] = self.cluster_num
            while len(neighbors) > 0:
                neighbor = int(neighbors.pop())
                sub_neighbors = self.get_neighbor(neighbor)
                if len(sub_neighbors) >= self.minpts:
                    for sub_neighbor in map(int, sub_neighbors):
                        if self.label[sub_neighbor] == UNCLASSFIED or self.label[sub_neighbor] == OUTLIER:
                            if self.label[sub_neighbor] == UNCLASSFIED:
                                neighbors.add(sub_neighbor)
                            self.label[sub_neighbor] = self.cluster_num
            return True

    def get_neighbor(self, idx):
        neighbors = set()
        for datum in self.data:
            if np.linalg.norm(self.data[idx][1:] - datum[1:]) <= self.eps:
                neighbors.add(datum[0])

        return neighbors


def main():
    argv = sys.argv

    if len(argv) == 5:
        input_file = argv[1]
        n = int(argv[2])
        eps = int(argv[3])
        minpts = int(argv[4])
    else:
        print("usage : python3", argv[0].split('/')[-1], "<cluster_num> <eps> <min_pts>")
        exit(0)

    file_number = input_file.split('.')[0][5:]
    output_file_format = 'input' + file_number + '_cluster_{}.txt'

    input_data = pd.read_csv(input_file, sep='\t', header=None).values
    model = DBSCAN(input_data, eps, minpts)
    model.fit()

    clusters = sorted(model.cluster.keys(), key=lambda x: len(model.cluster[x]), reverse=True)[:n]
    for num, c in enumerate(clusters):
        output_file_name = output_file_format.format(num)
        with open(output_file_name, 'w') as f:
            f.write('\n'.join(map(str, model.cluster[c])))


if __name__ == '__main__':
    main()
