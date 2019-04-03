
import numpy as np
from enum import Enum, unique

import time


# Status
UNKNOWN = -1
NOISE = -2

def timeit(func):
    def wrapper(*args, **wargs):
        start = time.time()
        res = func(*args, **wargs)
        end = time.time()
        print(f'{func.__name__}: {(end-start)*1000}')
        return res
    return wrapper


def load_data(path: str):
    with open(path, 'r') as f:
        data = []
        label = []
        for l in f.readlines():
            source = l.strip().split()
            data.append([float(val) for val in source[:2]])
            label.append(int(source[-1]))
        return np.array(data), np.array(label)


def get_dist(a, b, fast_mode: bool = False) -> float:
    """
    for float comparison, set all distance value precision to 5
    :param: a: np.array or np.matrix; record the coordinates of point
    :param: b: same as a
    :param: fast_mode: bool -> if True, ignore sqrt() opration for distance
    """

    if fast_mode:
        result = np.power(b-a, 2).sum()
    else:
        result = np.sqrt(np.power(b-a, 2).sum())
    return round(result, 5)


def get_neighbours(p: int, m, eps: float, fast_mode=False) -> list:
    """
    return neighbours index of given point p in source data matrix
    :param: p: int; index of given point in data matrix
    :param: m: np.matrix; N * 2 matrix recoding all nodes' coordinates
    """

    ngbs = []
    for idx in range(len(m)):
        if get_dist(m[p], m[idx, :], fast_mode) < eps:
            ngbs.append(idx)
    return ngbs


def clustering(p, m, eps, min_pts, tags, cluster_id, fast_mode=False) -> bool:

    neighbours = get_neighbours(p, m, eps, fast_mode)
    if len(neighbours) < min_pts:
        tags[p] = NOISE
        return False
    else:
        tags[p] = cluster_id
        for idx in neighbours:
            tags[idx] = cluster_id
        while len(neighbours) > 0:
            sub_neighbours = get_neighbours(neighbours[0], m, eps, fast_mode)
            if len(sub_neighbours) >= min_pts:
                for sub_n in sub_neighbours:
                    if tags[sub_n] < 0:
                        tags[sub_n] = cluster_id
                        if tags[sub_n] == UNKNOWN:
                            neighbours.append(sub_n)
                # neighbours.extend([sub_n for sub_n in sub_neighbours if tags[sub_n] == UNKNOWN])
            neighbours = neighbours[1:]
        return True


def naive_dbscan(m, eps, min_pts, fast_mode=False) -> list:
    """
    return list of labels as the sequence in data matrix
    :param: m: np.matrix; N * 2 matrix recoding all nodes' coordinates
    :param: eps: float; the value of radius of density area
    :param: min_pts: int; least neighbours should be in a density area
    """

    num_p = m.shape[0]
    tags = [UNKNOWN] * num_p

    cluster_id = 1
    for p_id in range(num_p):
        if tags[p_id] != UNKNOWN:
            continue
        if clustering(p_id, m, eps, min_pts, tags, cluster_id, fast_mode):
            cluster_id += 1
    return np.array(tags)


if __name__ == '__main__':
    path = './data/shape-sets/spiral.txt'
    m, y = load_data(path)
    labels = naive_dbscan(m, 2, 2)

    print(round((labels==y).sum()/len(labels), 4))

