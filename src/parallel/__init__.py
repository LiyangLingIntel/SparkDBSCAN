import numpy as np

from src.parallel.dbscan_rtree import rtree_partition
from src.parallel.dbscan_general import spatial_partition, local_dbscan, merge
from src.parallel.dbscan_general import load_data, load_data_label
from src.utils import timeit


def partition(dataset, eps, partition_tuple, rtree=False, rtree_mtd='cbs'):
    if rtree:
        partition_tuple = np.prod(partition_tuple)
        partitioned_rdd = rtree_partition(dataset,
                                          partition_tuple,
                                          eps,
                                          mtd=rtree_mtd)
    else:
        partitioned_rdd = spatial_partition(dataset, partition_tuple, eps)
    local_tags = partitioned_rdd.mapValues(lambda x: local_dbscan(x)).collect()
    result_tags = merge(local_tags, dataset)

    return result_tags


# parallel entry function
@timeit
def parallel_dbscan(dataset,
                    eps,
                    min_pts,
                    partition_tuple,
                    method='matrix',
                    metric='euclidian',
                    rtree=False,
                    rtree_mtd='cbs'):

    b_dataset = sc.broadcast(dataset)
    b_eps = sc.broadcast(eps)
    b_min_pts = sc.broadcast(min_pts)

    partitioned_rdd = partition(dataset,
                                eps,
                                partition_tuple,
                                rtree=rtree,
                                rtree_mtd=rtree_mtd)
    local_tags = partitioned_rdd.mapValues(lambda x: local_dbscan(
        x, method=method, metric=metric)).collect()
    result_tags = merge(local_tags, dataset)

    return result_tags
