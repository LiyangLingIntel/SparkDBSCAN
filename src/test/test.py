"""
Unit test or some module test can be added here
"""
import os

from src.utils import DataLoader, Evaluation
from src.serial import NaiveDBSCAN, MatrixDBSCAN

if __name__ == '__main__':
    src = '../data/shape-sets/r15_600.txt'
    dataset, _ = DataLoader.load_data_label(src)

    print('Naive DBSCAN:')
    ndbscan = NaiveDBSCAN(dataset)
    ndbscan.predict(0.7, 12)
    del ndbscan

    print('Matrix DBSCAN:')
    eps = [0.7, ]
    min_pts = [12, ]
    for i in eps:
        for j in min_pts:
            mdbscan = MatrixDBSCAN(dataset)
            mdbscan.predict(i, j)
            Evaluation.silhouette_coefficient(mdbscan)
            # print(mdbscan.tags)
            del mdbscan
