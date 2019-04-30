import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

from src.serial import NaiveDBSCAN, MatrixDBSCAN
from src.utils import DataLoader, Evaluation

import numpy as np

# Status
UNKNOWN = -1
NOISE = -2


def load_data_label(path):
    pts = sc.textFile(path).map(lambda x: x.strip().split()[:-1]).map(
        lambda x: tuple([float(i) for i in x]))
    return pts.collect()


def partition(dataset, n_partitions, eps):
    """
    :param: dataset: numpy array; 2-d numpy array or matrix which contains the coordinates of each point
    :param: n_patitions: int; number of partitions
    :param: eps: float; density distance threshold
    """

    # cut bins
    lower_bound = np.min(dataset, axis=0)
    upper_bound = np.max(dataset, axis=0)
    a = np.linspace(lower_bound, upper_bound, n_partitions, endpoint=False)
    b = np.array([upper_bound])
    tmp_bin = np.concatenate((a, b), axis=0)
    lower_bounds = [coordinates - eps for coordinates in tmp_bin[:-1]]
    upper_bounds = [coordinates + eps for coordinates in tmp_bin[1:]]
    print(lower_bounds)
    print(upper_bounds)

    # scatter points into bins with eps
    indexed_data = []
    for id_pts in range(len(dataset)):  # index of point in dataset
        for id_ptt in range(n_partitions):
            if not (dataset[id_pts] > lower_bounds[id_ptt]).all():
                continue
            if not (dataset[id_pts] < upper_bounds[id_ptt]).all():
                continue
            indexed_data.append([id_ptt, id_pts])

    res = sc.parallelize(
        indexed_data).groupByKey().map(lambda x: [x[0], list(x[1])])
    return res


def local_dbscan(partioned_rdd):
    #     rdd_data = [data for data in partioned_rdd]
    #     ids = [id_pts for id_pts in rdd_data[0]]
    dataset = np.array([b_dataset.value[idp] for idp in partioned_rdd])
    dbscan_obj = MatrixDBSCAN(dataset)
    dbscan_obj.predict(b_eps.value, b_min_pts.value)
    is_core_list = dbscan_obj._find_core_pts(b_eps.value, b_min_pts.value)

    return list(zip(zip(partioned_rdd, is_core_list), dbscan_obj.tags))


def merge(local_tags, dataset):
    global_tags = [UNKNOWN] * len(dataset)
    is_tagged = [0] * len(dataset)
    last_max_label = 0
    for local in local_tags:
        np_local = np.array(local[-1])
        np_local[:, -1] += last_max_label
        last_max_label = np.max(np_local[:, -1])

        # check and merge overlapped points
        tagged_indices = np.nonzero(is_tagged)[0]
        for tmp_i in range(len(np_local)):
            # should do tag check
            (p_id, is_core), label = np_local[tmp_i]
            if p_id in tagged_indices and is_core == 1:
                np_local[-1][np_local[-1] == label] = global_tags[p_id]

        # update global tags
        for (p_id, is_core), label in np_local:
            if is_tagged[p_id] == 1:
                continue
            global_tags[p_id] = label
            is_tagged[p_id] = 1
    return global_tags


if __name__ == '__main__':

    test_file = '../../data/shape-sets/r15_600.txt'
    dataset = load_data_label(test_file)
    n_partitions = 4
    eps = 0.7
    min_pts = 12

    b_dataset = sc.broadcast(dataset)
    b_eps = sc.broadcast(eps)
    b_min_pts = sc.broadcast(min_pts)

    partitioned_rdd = partition(dataset, n_partitions, eps)
    local_tags = partitioned_rdd.mapValues(lambda x: local_dbscan(x)).collect()
    result_tags = merge(local_tags, dataset)

    dbscan_obj = MatrixDBSCAN(dataset)
    dbscan_obj.tags = result_tags
    Evaluation.silhouette_coefficient(dbscan_obj)
