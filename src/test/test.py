"""
Unit test or some module test can be added here
"""
import os
print(os.getcwd())

from src.serial import NaiveDBSCAN, MatrixDBSCAN

if __name__ == '__main__':
    src = '../data/shape-sets/spiral.txt'

    print('Naive DBSCAN:')
    ndbscan = NaiveDBSCAN(src)
    ndbscan.predict(2.5, 3)
    del ndbscan

    print('Matrix DBSCAN:')
    mdbscan = MatrixDBSCAN(src)
    mdbscan.predict(2.5, 3)
    del mdbscan