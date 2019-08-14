# Spark Meets DBSCAN 
MSBD5001 Big Data Computing Projects -- Algorithm Parallelization.

[TOC]

### File Structure

Below is the tree of source code our parallel implementation of DBSCAN. 

```
src
├── ipynb
│   ├── parallel_dbscan.ipynb 
│   ├── MatrixDBSCAN.ipynb
│   ├── NaiveDBSCAN.ipynb
│   ├── partitioning.ipynb
│   ├── rtree_fixpartition.ipynb
│   └── rtree_partitioning.ipynb
├── parallel
│   ├── __init__.py
│   ├── dbscan_general.py
│   └── dbscan_rtree.py
├── serial
│   ├── __init__.py
│   └── dbscan.py
├── settings.py
├── test
│   ├── playground.py
│   └── test.py
└── utils
    ├── __init__.py
    ├── logger.py
    └── utils.py
```
Our python code is under folder `parallel`, `serial`, `test` and `utils` , and `ipynb` files include how we explore the way of implementing.

* `serial`: Under this folder, we implemented the serial DBSCAN algorithm and the improvement methods, and this module will act as local DBSCAN. 
* `parallel`: Under this folder, we implemented our parallel functions about partitioning, merging in `pyspark`. Two file `dbscan_general.py` and `dbscan_rtree.py` namely contain our strategy on spartial evenly split and two rtree-based partition. This entire folder also act as a module can be call outside.
* `utils`: In this module, we implemented utility functions, like clustering evaluation function and timer function.
* `settings.py`: Under this file, we set some of configuration and global status.
* `ipynb`: Under this folder, we used jupyter notebook to do some exploratory work，and plot out the clustering result. The final plots of our experiments mainly come from `parallel_dbscan.ipynb`.
* `test`: Under this folder, some test can be done here.

In `dbscan.py` two ways of serial dbscan algorithm is implemented: Naive method with redundant computation and optimal method with distance matrix.

Here I tested with **Spiral** dataset on [**Clustering basic benchmark**](http://cs.joensuu.fi/sipu/datasets/), which has 312 points of 2-degree and 3-cluster. Got time consuming in mini-second as below:

```pseudocode
Naive DBSCAN:
predict: 1886.1682415008545
Matrix DBSCAN:
predict: 2.608060836791992
```

It looks quite acceptable, it might be better in `neighbours map` as we planed, from the evidence of matrix method.

Further works will be on proper **evaluation** method and **parallel** implementation.

## Dues:

- **2019/03/18** **Proposal Integration**
- **2019/03/22** **Proposal Submission**
- **2019/04/07** **First Progress Check**
- **2019/04/14** **Overall Code Works Done**



## Notes

#### Reference:

* **Project Example:** [Implementation of DBSCAN Algorithm with python Spark APIs](https://www.cse.ust.hk/msbd5003/pastproj/deep1.pdf)
* [MR-DBSCAN: a scalable MapReduce-based DBSCAN algorithm for heavily skewed data](https://www.researchgate.net/publication/260523383_MR-DBSCAN_a_scalable_MapReduce-based_DBSCAN_algorithm_for_heavily_skewed_data)
* [Research on the Parallelization of the DBSCAN Clustering Algorithm for Spatial Data Mining Based on the Spark Platform](https://www.researchgate.net/publication/321753740_Research_on_the_Parallelization_of_the_DBSCAN_Clustering_Algorithm_for_Spatial_Data_Mining_Based_on_the_Spark_Platform)
* [DBSCAN Wiki](<https://en.wikipedia.org/wiki/DBSCAN>)

#### Data Source:

* [Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)

#### Key Points:

* http://localhost:4041/jobs/

  使用spark自带的任务监视器去查看任务的用时，资源分配，以及spark自动生成的DAG


## 1. Proposal

### 1.1. Requirements

The proposal of a deep project should contain the following contents:

1. Description of the problem.
2. Description of the algorithm (pseudo-code or language description).
3. Brief plan on how to implement it in Spark.

### 1.2. Draft

1. **Introduction** ---- **Liu Jinyu**

   1. *Problem Description*

2. **Sequential DBSCAN**  ---- **Wang Shen**

   1. *Algorithm explain*
   2. *Pseudo-code*
   3. *Figure illustration*

3. **Parallel DBSCAN** ---- **Ling Liyang**

   1. *Overview*

   2. *Plans on parallelisation* 

## 2. Task Arrangement

1. HDFS&Spark cluster deployment
2. Sequential DBSCAN efficiency test
3. Parallel DBSCAN
   1. Space based partition
      1. Naïve redundant computation method
      2. Distance Matrix
      3. Neighbour List
   2. Cost based partition

## 3. Presentation

### 3.1. Requirements 

1. Description of the problem.
2. Description of the algorithm (pseudo-code or language description).
3. How you implemented the algorithm in Spark, including any optimizations that you have done.
4. Experimental results, which should demonstrate the scalability of your implementation as the data size and the number of executors grows.
5. Potential improvements, if any.

### 3.2 Pipeline

* **Problem Introduction** — 30s
* **Local DBSCAN Description** — 2min
* **Implementation in Spark** — 6 min
  * **General implementation** — 2 min
    * **Evenly partition**
    * **Merging**
  * **Optimizations** — 4 min
    * **Improvement on local DBSCAB** — 1min
      - **distance matrix**
      - **adjacent list**
    * **Improvements on Partition** — 3 min
      * **RTree: Cost-based** 
      * **RTree: Reduced boundary**
* **Experimental results** — 4 min
  * **Brief introduction on how to tuning hyper parameters**
  * **Efficiency  with different data distributions**
  * **Comparation of above implemetations on each dataset**
* **Summary and Further work** — 30s

## 4. Final Report

### 4.1. Requirements

1. The source code of your implementation (as a separate file).
2. References (including others' implementation of the same algorithm).