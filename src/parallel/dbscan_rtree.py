from rtree import index
from collections import deque
from queue import PriorityQueue
import math
import numpy as np

import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))


def _get_cost(bounds, nPoints, fanout=2):
    h = math.log((nPoints + 1) / fanout, fanout) + 1
    DA = h + math.sqrt(nPoints) * 2 / (math.sqrt(fanout) -
                                       1) + nPoints / (fanout - 1) + 1
    return DA * nPoints


def cost_base_partition(rtree, maxCost, eps):
    mbr = rtree.bounds
    partition_list = []
    queue = deque()
    queue.append(mbr)
    while len(queue):
        br = queue.popleft()
        nPoints = rtree.count(br)
        if get_cost(br, nPoints) > maxCost:
            (subbr1, subbr2) = _cost_base_split(rtree, br, eps)
            queue.append(subbr1)
            queue.append(subbr2)
        else:
            partition_list.append(br)
    return partition_list


def _cost_base_split(rtree, bounds, eps):
    (xmin, ymin, xmax, ymax) = bounds
    #vertical split
    ymin_diff = float('inf')
    ysplit = ymin + (ymax - ymin) / 2
    ybest_split = ((xmin, ymin, xmax, ysplit), (xmin, ysplit, xmax, ymax))
    while (ysplit + eps * 2 <= ymax):
        lowerbr = (xmin, ymin, xmax, ysplit)
        lowercost = _get_cost(lowerbr, rtree.count(lowerbr))

        upperbr = (xmin, ysplit, xmax, ymax)
        uppercost = _get_cost(upperbr, rtree.count(upperbr))
        costdiff = abs(uppercost - lowercost)
        if costdiff < ymin_diff:
            ymin_diff = costdiff
            ybest_split = (lowerbr, upperbr)
            if uppercost < lowercost:
                ysplit = ymin + (ysplit - ymin) / 2
            else:
                ysplit = ysplit + (ymax - ysplit) / 2
        else:
            break

    #horizontal split
    xmin_diff = float('inf')
    xsplit = xmin + (xmax - xmin) / 2
    xbest_split = ((xmin, ymin, xsplit, ysplit), (xsplit, ymin, xmax, ymax))
    while (xsplit + eps * 2 <= xmax):
        lowerbr = (xmin, ymin, xsplit, ymax)
        lowercost = _get_cost(lowerbr, rtree.count(lowerbr))

        upperbr = (xsplit, ymin, xmax, ymax)
        uppercost = _get_cost(upperbr, rtree.count(upperbr))
        costdiff = abs(uppercost - lowercost)
        if costdiff < xmin_diff:
            xmin_diff = costdiff
            xbest_split = (lowerbr, upperbr)
            if uppercost < lowercost:
                xsplit = xmin + (xsplit - xmin) / 2
            else:
                xsplit = xsplit + (xmax - xsplit) / 2
        else:
            break

    #compare ysplit and xsplit
    if xmin_diff < ymin_diff:
        return xbest_split
    else:
        return ybest_split


def reduced_boundary_partition(rtree, maxPoints, eps):
    mbr = rtree.bounds
    partition_list = []
    queue = deque()
    queue.append(mbr)
    while len(queue):
        br = queue.popleft()
        nPoints = rtree.count(br)
        if nPoints > maxPoints:
            (br1, br2) = _reduced_boundary_split(rtree, br, eps)
            queue.append(br1)
            queue.append(br2)
        else:
            partition_list.append(br)
    return partition_list


def _reduced_boundary_split(rtree, br, eps):

    (xmin, ymin, xmax, ymax) = br

    #vertical splitline candidates
    ymin_score = float('inf')
    ysplit = ymin + (ymax - ymin) / 2
    ybest_split = ((xmin, ymin, xmax, ysplit), (xmin, ysplit, xmax, ymax))
    while (ysplit + eps * 2 <= ymax):
        br1 = (xmin, ymin, xmax, ysplit)
        br2 = (xmin, ysplit, xmax, ymax)
        point_diff = abs(rtree.count(br1) - rtree.count(br2))
        score = point_diff * rtree.count(
            (xmin, ysplit - eps, xmax, ysplit + eps))
        if score < ymin_score:
            ymin_score = score
            ybest_split = (br1, br2)
            if rtree.count(br1) > rtree.count(br2):
                ysplit = ymin + (ysplit - ymin) / 2
            else:
                ysplit = ysplit + (ymax - ysplit) / 2
        else:
            break

    #horizontal splitline candidates
    xsplit = xmin + eps * 2
    xmin_score = float('inf')
    xbest_split = ((xmin, ymin, xsplit, ymax), (xsplit, ymin, xmax, ymax))
    while (xsplit + eps * 2 <= xmax):
        br1 = (xmin, ymin, xsplit, ymax)
        br2 = (xsplit, ymin, xmax, ymax)
        point_diff = abs(rtree.count(br1) - rtree.count(br2))
        score = point_diff * rtree.count((xmin - eps, ymin, xmin + eps, ymax))
        if score < xmin_score:
            xmin_score = score
            xbest_split = (br1, br2)
            if rtree.count(br1) > rtree.count(br2):
                xsplit = xmin + (xsplit - xmin) / 2
            else:
                xsplit = xsplit + (xmax - xsplit) / 2
        else:
            break

    if xmin_score < ymin_score:
        return xbest_split
    else:
        return ybest_split


#construct rtree index
def construct_rtree_index(dataset):
    p = index.Property()
    rtree_idx = index.Index(properties=p)
    count = 0
    for coordinate in dataset:
        rtree_idx.insert(count, (*coordinate, *coordinate))
        count += 1
    return rtree_idx


def cbs_fixnum(rtree, eps, n_partition):
    mbr = rtree.bounds
    partition_list = []
    queue = PriorityQueue()
    cost = _get_cost(mbr, rtree.count(mbr))
    queue.put((-cost, mbr))
    while queue.qsize() < n_partition:
        (score, br) = queue.get()
        subbrs = _cost_base_split(rtree, br, eps)
        for subbr in subbrs:
            cost = _get_cost(subbr, rtree.count(subbr))
            queue.put((-cost, subbr))
    while queue.qsize() > 0:
        (score, br) = queue.get()
        partition_list.append(br)
    return partition_list


def rbs_fixnum(rtree, eps, n_partition):
    mbr = rtree.bounds
    partition_list = []
    queue = PriorityQueue()
    queue.put((-rtree.count(mbr), mbr))
    while queue.qsize() < n_partition:
        (score, br) = queue.get()
        subbrs = _reduced_boundary_split(rtree, br, eps)
        for subbr in subbrs:

            queue.put((-rtree.count(subbr), subbr))
    while queue.qsize() > 0:
        (score, br) = queue.get()
        partition_list.append(br)
    return partition_list


def rtree_partition(dataset, n_patitions, eps, mtd='cbs'):
    """
    params:parameters needed for different spliting method; for cost-base method params=('cbs', max_cost);
    for reduced-boundary method params = ('rbs', max_points)
    """
    idx = construct_rtree_index(dataset)
    # #split test
    if mtd == 'cbs':
        partitioned = cbs_fixnum(idx, eps, n_patitions)
    else:
        partitioned = rbs_fixnum(idx, eps, n_patitions)

    indexed_data = []
    id_ptt = 0
    for boundary in partitioned:
        (left, bot, right, top) = boundary
        for id_pts in idx.intersection(
            (left - eps, bot - eps, right + eps, top + eps)):
            indexed_data.append([id_ptt, id_pts])
        id_ptt += 1

    res = sc.parallelize(
        indexed_data).groupByKey().map(lambda x: [x[0], list(x[1])])
    return res
