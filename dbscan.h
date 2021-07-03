#ifndef DBSCAN_H
#define DBSCAN_H

#include "common.h"

bool MonitorSeedPoints(vector<int>& unprocessedPoints, int* runningCluster,
                       int* d_cluster, int* d_seedList, int* d_seedLength,
                       int* d_collisionMatrix, int* d_extraCollision,
                       int* d_results, float *mergeTime);

void GetDbscanResult(int* d_cluster, int* runningCluster, int* clusterCount,
                     int* noiseCount);

__global__ void DBSCAN(double* dataset, int* cluster, int* seedList,
                       int* seedLength, int* collisionMatrix,
                       int* extraCollision, int* results,
                       struct IndexStructure** indexBuckets,

                       int* indexesStack, int* dataValue, double* upperBounds,
                       double* binWidth);

__device__ void MarkAsCandidate(int neighborID, int chainID, int* cluster,
                                int* seedList, int* seedLength,
                                int* collisionMatrix, int* extraCollision);

// TESTING
__global__ void DBSCAN_ONE_INSTANCE(double* dataset, int* cluster,
                                    int* seedList, int* seedLength,
                                    int* collisionMatrix, int* extraCollision,
                                    int* results,
                                    struct IndexStructure** indexBuckets,
                                    int* indexesStack, int* dataValue,
                                    double* upperBounds, double* binWidth);

__global__ void COLLISION_DETECTION(int* collisionMatrix, int* extraCollision,
                                    int* cluster, int* clusterMap,
                                    int* clusterCountMap, int* runningCluster);

__global__ void COLLISION_MERGE(int* collisionMatrix, int* extraCollision,
                                int* cluster, int* clusterMap,
                                int* clusterCountMap);

void TestGetDbscanResult(int* d_cluster, int* runningCluster, int* clusterCount,
                         int* noiseCount);

bool TestMonitorSeedPoints(vector<int>& unprocessedPoints, int* d_cluster,
                           int* d_seedList, int* d_seedLength,
                           int* d_collisionMatrix, int* d_extraCollision,
                           int* d_results, int* d_clusterMap,
                           int* d_clusterCountMap, int* d_runningCluster);

void searchFromIndexTree(int* d_cluster, double* d_upperBounds, double* dataset,
                         int* d_seedList, int* d_seedLength, int* d_results,
                         int indexTreeMetaData[TREE_LEVELS * RANGE]);

#endif