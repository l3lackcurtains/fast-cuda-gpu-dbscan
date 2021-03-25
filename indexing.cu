
#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <map>
#include <math.h>
#include <set>
#include <vector>

#include "common.h"
#include "indexing.h"

__device__ void indexConstruction(int level, int *indexTreeMetaData,
                                  double *minPoints, double *binWidth,
                                  struct IndexStructure **indexBuckets,
                                  double *upperBounds) {
  for (int k = threadIdx.x + indexTreeMetaData[level * RANGE + 0];
       k < indexTreeMetaData[level * RANGE + 1]; k = k + THREAD_COUNT) {
    for (int i = 0; i < PARTITION_SIZE; i++) {
      int currentBucketIndex =
          indexTreeMetaData[level * RANGE + 1] + i +
          (k - indexTreeMetaData[level * RANGE + 0]) * PARTITION_SIZE;

      indexBuckets[k]->dimension = level;
      indexBuckets[currentBucketIndex]->dimension = level + 1;

      if (i == 0) {
        indexBuckets[k]->childFrom = currentBucketIndex;
      }

      double rightPoint =
          minPoints[level] + i * binWidth[level] + binWidth[level];

      if (i == PARTITION_SIZE - 1) rightPoint = rightPoint + binWidth[level];

      upperBounds[currentBucketIndex] = rightPoint;
    }
  }
  __syncthreads();
}

__device__ void insertData(int id, double *dataset,
                           struct IndexStructure **indexBuckets, int *dataKey,
                           int *dataValue, double *upperBounds,
                           double *binWidth) {
  double data[DIMENSION];
  for (int j = 0; j < DIMENSION; j++) {
    data[j] = dataset[id * DIMENSION + j];
  }

  int currentIndex = 0;
  bool found = false;

  while (!found) {
    if (indexBuckets[currentIndex]->dimension >= DIMENSION) break;
    double comparingData = data[indexBuckets[currentIndex]->dimension];

    int k = thrust::upper_bound(thrust::device, upperBounds + indexBuckets[currentIndex]->childFrom,
      upperBounds + indexBuckets[currentIndex]->childFrom + PARTITION_SIZE, comparingData, thrust::less<double>()) - upperBounds;

    if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {
      dataValue[id] = id;
      dataKey[id] = k;
      found = true;
    }
    currentIndex = k;      
  }
}


__global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                                   double *minPoints, double *binWidth,
                                   int *results,
                                   struct IndexStructure **indexBuckets,
                                   int *dataKey, int *dataValue,
                                   double *upperBounds) {
  if (blockIdx.x < DIMENSION) {
    indexConstruction(blockIdx.x, indexTreeMetaData, minPoints, binWidth,
                      indexBuckets, upperBounds);
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < DATASET_COUNT;
       i = i + THREAD_COUNT * THREAD_BLOCKS) {
    insertData(i, dataset, indexBuckets, dataKey, dataValue, upperBounds,
               binWidth);
  }
  __syncthreads();
}

__global__ void INDEXING_ADJUSTMENT(int *indexTreeMetaData,
                                    struct IndexStructure **indexBuckets,
                                    int *dataKey) {
  __shared__ int indexingRange;
  if (threadIdx.x == 0) {
    indexingRange = indexTreeMetaData[DIMENSION * RANGE + 1] -
                    indexTreeMetaData[DIMENSION * RANGE];
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = threadId; i < indexingRange;
       i = i + THREAD_COUNT * THREAD_BLOCKS) {
    int idx = indexTreeMetaData[DIMENSION * RANGE] + i;

    thrust::pair<int *, int *> dataPositioned;

    dataPositioned = thrust::equal_range(thrust::device, dataKey, dataKey + DATASET_COUNT, idx);

    indexBuckets[idx]->dataBegin = dataPositioned.first - dataKey;
    indexBuckets[idx]->dataEnd = dataPositioned.second - dataKey;
  }
  __syncthreads();
}
