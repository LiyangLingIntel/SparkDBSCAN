"""
Unit test or some module test can be added here
"""
import os
print(os.getcwd())

from src.utils import DataLoader, Evaluation
from src.serial import NaiveDBSCAN, MatrixDBSCAN

if __name__ == '__main__':
    src = '../data/shape-sets/r15_600.txt'

    # print('Naive DBSCAN:')
    # ndbscan = NaiveDBSCAN(src)
    # ndbscan.predict(2.5, 3)
    # del ndbscan

    print('Matrix DBSCAN:')
    eps = [0.6, ]
    min_pts = [12, ]
    dataset, _ = DataLoader.load_data_label(src)

    for i in eps:
        for j in min_pts:
            mdbscan = MatrixDBSCAN(dataset)
            mdbscan.predict(i, j)
            Evaluation.silhouette_coefficient(mdbscan)

            print(mdbscan.tags)
            del mdbscan
