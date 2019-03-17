# Spark Meets DBSCAN 
MSBD5001 Big Data Computing Projects -- Algorithm Parallelization.

[TOC]

#### Reference:

* **Project Example:** [Implementation of DBSCAN Algorithm with python Spark APIs](https://www.cse.ust.hk/msbd5003/pastproj/deep1.pdf)
* [MR-DBSCAN: a scalable MapReduce-based DBSCAN algorithm for heavily skewed data](https://www.researchgate.net/publication/260523383_MR-DBSCAN_a_scalable_MapReduce-based_DBSCAN_algorithm_for_heavily_skewed_data)
* [Research on the Parallelization of the DBSCAN Clustering Algorithm for Spatial Data Mining Based on the Spark Platform](https://www.researchgate.net/publication/321753740_Research_on_the_Parallelization_of_the_DBSCAN_Clustering_Algorithm_for_Spatial_Data_Mining_Based_on_the_Spark_Platform)
* [DBSCAN Wiki](<https://en.wikipedia.org/wiki/DBSCAN>)

#### Data Source:

* [Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)

#### Winning Points:

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

   1. Overview

      From the description of sequencial DBSCAN, we know that common DBSCAN is doing works as searching core points, calculating neighbours through distances and giving cluster tags among all given data points recurrently. Fortunately the distance calculation has little relationships among different point pairs which makes it possiple to implement DBSCAN algorithm concurrently. 

      In theory, algorithm efficiency will be improved if the complete dataset can be diveded into multiple independent small data groups, and the points traverse and distance calculation process can be done by several processors. So, the main tasks should we focus on are dataset partitioning and cluster merging.

   2. Plan on parallelisation 

      Before getting started, as most parallel programs, there are two core problems should have the proper solutions.

      First one is data synchronisation. More specifically, we should define a effective way to deal with the points on the boundary of different partitions, in that case, clusters on different partitions could have the posibility to form into the same cluster in the merging process.

      Secondly, the effiency is another key point should be put into consideration. Because based on the first problem, extra calculation is needed for patitioning and communication among partitions. What's more, the workloads and efficiency of each worker should also be considered, or some workers' idle and some workers' with heavy work which is not a ideal situation in parallel programs.

      So, for DBSCAN parallel implementation, we have initially steps as below.

      <img src="https://ws2.sinaimg.cn/large/006tKfTcgy1g1681smic1j30si0wwwhb.jpg" width="450" />

      * Load data from HDFS, preprocess to get input dataset
      * Analyze dataset to get info about data size and distribution, based on it, allocate data points into partitions through proper methods
      * Do local DBSCAN on each partition
      * Merge the clustering results of partitions, remark to get final results

      In the last stage, datasets with different size and data distribution will be applied to test the efficiency of parallelisation and help to improve our algorithm.

### 1.3. Due Dates

* **2019/03/18** **Integration** ---- **Song Hongzhen** & **Liu Jinyu**

* **2019/03/22** **Submission**



## 2. Presentation

### 2.1. Requirements 

1. Description of the problem.
2. Description of the algorithm (pseudo-code or language description).
3. How you implemented the algorithm in Spark, including any optimizations that you have done.
4. Experimental results, which should demonstrate the scalability of your implementation as the data size and the number of executors grows.
5. Potential improvements, if any.



## 3. Final Report

### 3.1. Requirements

1. The source code of your implementation (as a separate file).
2. References (including others' implementation of the same algorithm).