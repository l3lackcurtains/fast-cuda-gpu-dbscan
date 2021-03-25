#ifndef DBSCAN_H
#define DBSCAN_H

#include "common.h"
#include "dbscan.h"

bool MonitorSeedPoints(vector<int> &unprocessedPoints,
                       map<int, set<int>> &collisionUnion, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_results);

void GetDbscanResult(int *d_cluster, int *runningCluster, int *clusterCount,
                     int *noiseCount);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,

                       int *indexesStack, int *dataValue, double *upperBounds,
                       double *binWidth);

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *collisionMatrix, int *extraCollision);



#endif