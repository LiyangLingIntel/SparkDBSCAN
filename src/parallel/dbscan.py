import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

from src.serial import NaiveDBSCAN, MatrixDBSCAN
from src.utils import DataLoader, Evaluation
from src.settings import UNKNOWN, NOISE

import numpy as np

# broadcast variable
b_dataset = None
b_eps = None
b_min_pts = None


def load_data_label(path):
    pts = sc.textFile(path).map(lambda x: x.strip().split()[:-1]).map(
        lambda x: tuple([float(i) for i in x]))
    return pts.collect()


def _bounds_coordinates(bin_bounds):
    #     coordinates = [bounds[0]]

    lower_cdnts = [[low] for low in bin_bounds[0][:-1]]
    upper_cdnts = [[high] for high in bin_bounds[0][1:]]

    # super stupid implementation, optimization needed
    for bound in bin_bounds[1:]:
        lower_tmp = []
        upper_tmp = []

        for bc in bound[:-1]:
            lower_tmp.extend([lc + [bc] for lc in lower_cdnts])
        lower_cdnts = lower_tmp

        for bc in bound[1:]:
            upper_tmp.extend([uc + [bc] for uc in upper_cdnts])
        upper_cdnts = upper_tmp

    return np.array(lower_cdnts), np.array(upper_cdnts)


def partition(dataset, n_partitions, eps):
    """
    :param: dataset: numpy array; 2-d numpy array or matrix which contains the coordinates of each point
    :param: n_patitions: tuple; (x, y, ...) product of elements in this tuple which has size corresponds to data dimensions
    :param: eps: float; density distance threshold
    """
    partition_dim = n_partitions
    n_partitions = np.prod(n_partitions)

    # cut bins
    bounds = np.concatenate(
        ([np.min(dataset, axis=0)], [np.max(dataset, axis=0)]),
        axis=0)  # 2 * D
    bounds_dim = bounds.T  # D * 2,

    bin_bounds = []
    for i in range(len(partition_dim)):
        dim_bins = np.linspace(*bounds_dim[i],
                               partition_dim[i] + 1,
                               endpoint=True)
        bin_bounds.append(dim_bins)

    lower_bounds, upper_bounds = _bounds_coordinates(bin_bounds)
    lower_bounds -= eps
    upper_bounds += eps

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


# parallel entry function
def parallel_dbscan(dataset, eps, min_pts, partition_tuple):
    b_dataset = sc.broadcast(dataset)
    b_eps = sc.broadcast(eps)
    b_min_pts = sc.broadcast(min_pts)

    partitioned_rdd = partition(dataset, n_partitions, eps)
    local_tags = partitioned_rdd.mapValues(lambda x: local_dbscan(x)).collect()
    result_tags = merge(local_tags, dataset)

    return result_tags


if __name__ == '__main__':

    test_file = '../../data/shape-sets/r15_600.txt'
    dataset = load_data_label(test_file)
    n_partitions = (4, 2)
    eps = 0.7
    min_pts = 12

    result_tags = parallel_dbscan(dataset, eps, min_pts, n_partitions)

    dbscan_obj = MatrixDBSCAN(np.array(dataset))
    dbscan_obj.tags = result_tags
    dbscan_obj.eps = eps
    dbscan_obj.min_pts = min_pts

    Evaluation.silhouette_coefficient(dbscan_obj)