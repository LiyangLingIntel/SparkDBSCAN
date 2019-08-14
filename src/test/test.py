"""
Unit test or some module test can be added here
"""
import os
import random

from src.utils import DataLoader, Evaluation
from src.serial import NaiveDBSCAN, MatrixDBSCAN

if __name__ == '__main__':
    src = '../data/shape-sets/spiral.txt'
    dataset, _ = DataLoader.load_data_label(src)

    # print('Naive DBSCAN:')
    # ndbscan = NaiveDBSCAN(dataset, 0.7, 12)
    # ndbscan.predict()
    # del ndbscan

    # print('Matrix DBSCAN:')
    # eps = [2.5, 4]
    # min_pts = [15, 30]
    # for t in range(10):
    #     try:
    #         i = round(random.uniform(*eps), 2)
    #         j = random.randint(*min_pts)
    #         mdbscan = MatrixDBSCAN(dataset, i, j)
    #         mdbscan.predict()
    #         Evaluation.silhouette_coefficient(mdbscan)
    #         # print(mdbscan.tags)
    #         del mdbscan
    #     except Exception as e:
    #         print(e)
    
    print('Matrix DBSCAN:')
    eps = 3.5
    min_pts = 2
    mdbscan = MatrixDBSCAN(dataset, eps, min_pts,  metric='manhattan')
    mdbscan.predict()
    Evaluation.silhouette_coefficient(mdbscan)

