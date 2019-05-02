"""
Unit test or some module test can be added here
"""
import os

from src.utils import DataLoader, Evaluation
from src.serial import NaiveDBSCAN, MatrixDBSCAN

if __name__ == '__main__':
    src = '../data/shape-sets/pathbased_300.txt'
    dataset, _ = DataLoader.load_data_label(src)

    # print('Naive DBSCAN:')
    # ndbscan = NaiveDBSCAN(dataset, 0.7, 12)
    # ndbscan.predict()
    # del ndbscan

    print('Matrix DBSCAN:')
    eps = [0.6, 0.8, 1]
    min_pts = [1, 2, 3]
    for i in eps:
        for j in min_pts:
            mdbscan = MatrixDBSCAN(dataset, i, j)
            mdbscan.predict()
            Evaluation.silhouette_coefficient(mdbscan)
            # print(mdbscan.tags)
            del mdbscan
