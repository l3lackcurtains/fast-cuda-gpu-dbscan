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
#include "dbscan.h"

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,

                       int *indexesStack, int *dataValue, double *upperBounds,
                       double *binWidth) {
  // Point ID to expand by a block
  __shared__ int pointID;

  // Neighbors to store of neighbors points exceeds minpoints
  __shared__ int neighborBuffer[MINPTS];

  // It counts the total neighbors
  __shared__ int neighborCount;

  // ChainID is basically blockID
  __shared__ int chainID;

  // Store the point from pointID
  __shared__ double point[DIMENSION];

  // Length of the seedlist to check its size
  __shared__ int currentSeedLength;

  __shared__ int resultId;

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
    currentSeedLength = seedLength[chainID];
    pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int x = threadId; x < THREAD_BLOCKS * THREAD_BLOCKS;
       x = x + THREAD_BLOCKS * THREAD_COUNT) {
    collisionMatrix[x] = UNPROCESSED;
  }
  for (int x = threadId; x < THREAD_BLOCKS * EXTRA_COLLISION_SIZE;
       x = x + THREAD_BLOCKS * THREAD_COUNT) {
    extraCollision[x] = UNPROCESSED;
  }

  __syncthreads();

  // Complete the seedlist to proceed.

  while (seedLength[chainID] != 0) {
    for (int x = threadId; x < THREAD_BLOCKS * POINTS_SEARCHED;
         x = x + THREAD_BLOCKS * THREAD_COUNT) {
      results[x] = UNPROCESSED;
    }
    __syncthreads();

    // Assign chainID, current seed length and pointID
    if (threadIdx.x == 0) {
      chainID = blockIdx.x;
      currentSeedLength = seedLength[chainID];
      pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
    }
    __syncthreads();

    // Check if the point is already processed
    if (threadIdx.x == 0) {
      seedLength[chainID] = currentSeedLength - 1;
      neighborCount = 0;
      for (int x = 0; x < DIMENSION; x++) {
        point[x] = dataset[pointID * DIMENSION + x];
      }
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////

    searchPoints(point, chainID, dataset, results, indexBuckets, indexesStack,
                 dataValue, upperBounds, binWidth);

    __syncthreads();

    for (int k = 0; k < POINTS_SEARCHED; k++) {
      if (threadIdx.x == 0) {
        resultId = results[chainID * POINTS_SEARCHED + k];
      }
      __syncthreads();

      if (resultId == -1) break;

      for (int i = threadIdx.x + indexBuckets[resultId]->dataBegin;
           i < indexBuckets[resultId]->dataEnd; i = i + THREAD_COUNT) {
        register double comparingPoint[DIMENSION];

        for (int x = 0; x < DIMENSION; x++) {
          comparingPoint[x] = dataset[dataValue[i] * DIMENSION + x];
        }

        register double distance = 0;
        for (int x = 0; x < DIMENSION; x++) {
          distance +=
              (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
        }

        if (distance <= EPS * EPS) {
          register int currentNeighborCount = atomicAdd(&neighborCount, 1);
          if (currentNeighborCount >= MINPTS) {
            MarkAsCandidate(dataValue[i], chainID, cluster, seedList,
                            seedLength, collisionMatrix, extraCollision);
          } else {
            neighborBuffer[currentNeighborCount] = dataValue[i];
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////

    if (neighborCount >= MINPTS) {
      cluster[pointID] = chainID;
      for (int i = threadIdx.x; i < MINPTS; i = i + THREAD_COUNT) {
        MarkAsCandidate(neighborBuffer[i], chainID, cluster, seedList,
                        seedLength, collisionMatrix, extraCollision);
      }
    } else {
      cluster[pointID] = NOISE;
    }

    __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////

    if (threadIdx.x == 0 && seedLength[chainID] >= MAX_SEEDS) {
      seedLength[chainID] = MAX_SEEDS - 1;
    }
    __syncthreads();
  }
}

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_results) {
  int *localSeedLength;
  localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localSeedLength, d_seedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int *localSeedList;
  localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

  int *localCollisionMatrix;
  localCollisionMatrix =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyDeviceToHost));

  int *localExtraCollision;
  localExtraCollision =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE);
  gpuErrchk(cudaMemcpy(localExtraCollision, d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE,
                       cudaMemcpyDeviceToHost));

  ////////////////////////////////////////////////////////////////////////////////////////

  int clusterMap[THREAD_BLOCKS];
  set<int> blockSet;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    blockSet.insert(i);
  }

  set<int>::iterator it;

  while (blockSet.empty() == 0) {
    it = blockSet.begin();
    int curBlock = *it;
    set<int> expansionQueue;
    set<int> finalQueue;

    expansionQueue.insert(curBlock);
    finalQueue.insert(curBlock);

    while (expansionQueue.empty() == 0) {
      it = expansionQueue.begin();
      int expandBlock = *it;
      expansionQueue.erase(it);
      blockSet.erase(expandBlock);
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;
        if (localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    }

    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      clusterMap[*it] = curBlock;
    }
  }

  int clusterCountMap[THREAD_BLOCKS];
  for (int x = 0; x < THREAD_BLOCKS; x++) {
    clusterCountMap[x] = UNPROCESSED;
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    if (clusterCountMap[clusterMap[x]] != UNPROCESSED) continue;
    clusterCountMap[clusterMap[x]] = (*runningCluster);
    (*runningCluster)++;
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    thrust::replace(thrust::device, d_cluster, d_cluster + DATASET_COUNT, x,
                    clusterCountMap[clusterMap[x]]);
  }

  for (int x = 0; x < THREAD_BLOCKS; x++) {
    if (localExtraCollision[x * EXTRA_COLLISION_SIZE] == -1) continue;
    int minCluster = localExtraCollision[x * EXTRA_COLLISION_SIZE];
    thrust::replace(thrust::device, d_cluster, d_cluster + DATASET_COUNT,
                    clusterCountMap[clusterMap[x]], minCluster);
    for (int y = 0; y < EXTRA_COLLISION_SIZE; y++) {
      if (localExtraCollision[x * EXTRA_COLLISION_SIZE + y] == UNPROCESSED)
        break;
      int data = localExtraCollision[x * EXTRA_COLLISION_SIZE + y];
      thrust::replace(thrust::device, d_cluster, d_cluster + DATASET_COUNT,
                      data, minCluster);
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  int complete = 0;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    bool found = false;
    while (!unprocessedPoints.empty()) {
      int lastPoint = unprocessedPoints.back();
      unprocessedPoints.pop_back();

      if (localCluster[lastPoint] == UNPROCESSED) {
        localSeedLength[i] = 1;
        localSeedList[i * MAX_SEEDS] = lastPoint;
        found = true;
        break;
      }
    }

    if (!found) {
      complete++;
    }
  }

  // FInally, transfer back the CPU memory to GPU and run DBSCAN process

  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  // Free CPU memories

  free(localCluster);
  free(localSeedList);
  free(localSeedLength);
  free(localCollisionMatrix);
  free(localExtraCollision);

  if (complete == THREAD_BLOCKS) {
    return true;
  }

  return false;
}

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *collisionMatrix, int *extraCollision) {
  register int oldState =
      atomicCAS(&(cluster[neighborID]), UNPROCESSED, chainID);

  if (oldState == UNPROCESSED) {
    register int sl = atomicAdd(&(seedLength[chainID]), 1);
    if (sl < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + sl] = neighborID;
    }
  }

  else if (oldState >= THREAD_BLOCKS) {
    for (int i = 0; i < EXTRA_COLLISION_SIZE; i++) {
      register int changedState =
          atomicCAS(&(extraCollision[chainID * EXTRA_COLLISION_SIZE + i]),
                    UNPROCESSED, oldState);
      if (changedState == UNPROCESSED || changedState == oldState) {
        break;
      }
    }
  }

  else if (oldState != NOISE && oldState != chainID &&
           oldState < THREAD_BLOCKS) {
    collisionMatrix[oldState * THREAD_BLOCKS + chainID] = 1;
    collisionMatrix[chainID * THREAD_BLOCKS + oldState] = 1;
  }

  else if (oldState == NOISE) {
    oldState = atomicCAS(&(cluster[neighborID]), NOISE, chainID);
  }
}

__device__ void searchPoints(double *data, int chainID, double *dataset,
                             int *results, struct IndexStructure **indexBuckets,
                             int *indexesStack, int *dataValue,
                             double *upperBounds, double *binWidth) {
  __shared__ int resultsCount;
  __shared__ int indexBucketSize;
  __shared__ int currentIndex;
  __shared__ int currentIndexSize;
  __shared__ double comparingData;

  if (threadIdx.x == 0) {
    resultsCount = 0;
    indexBucketSize = 1;
    for (int i = 0; i < DIMENSION; i++) {
      indexBucketSize *= 3;
    }
    indexBucketSize = indexBucketSize * chainID;
    currentIndexSize = indexBucketSize;
    indexesStack[currentIndexSize++] = 0;
  }
  __syncthreads();

  while (currentIndexSize > indexBucketSize) {
    if (threadIdx.x == 0) {
      currentIndexSize = currentIndexSize - 1;
      currentIndex = indexesStack[currentIndexSize];
      comparingData = data[indexBuckets[currentIndex]->dimension];
    }
    __syncthreads();

    for (int k = threadIdx.x + indexBuckets[currentIndex]->childFrom;
         k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE;
         k = k + THREAD_COUNT) {
      double leftRange;
      double rightRange;
      if (k == indexBuckets[currentIndex]->childFrom) {
        leftRange =
            upperBounds[k] - binWidth[indexBuckets[currentIndex]->dimension];
      } else {
        leftRange = upperBounds[k - 1];
      }

      rightRange = upperBounds[k];

      if (comparingData >= leftRange && comparingData < rightRange) {
        if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {
          int oldResultsCount = atomicAdd(&resultsCount, 1);
          results[chainID * POINTS_SEARCHED + oldResultsCount] = k;

          if (k > indexBuckets[currentIndex]->childFrom) {
            oldResultsCount = atomicAdd(&resultsCount, 1);
            results[chainID * POINTS_SEARCHED + oldResultsCount] = k - 1;
          }

          if (k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE - 1) {
            oldResultsCount = atomicAdd(&resultsCount, 1);
            results[chainID * POINTS_SEARCHED + oldResultsCount] = k + 1;
          }
        } else {
          int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
          indexesStack[oldCurrentIndexSize] = k;
          if (k > indexBuckets[currentIndex]->childFrom) {
            int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
            indexesStack[oldCurrentIndexSize] = k - 1;
          }
          if (k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE - 1) {
            int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
            indexesStack[oldCurrentIndexSize] = k + 1;
          }
        }
      }
    }

    __syncthreads();
  }
}

void GetDbscanResult(int *d_cluster, int *runningCluster, int *clusterCount,
                     int *noiseCount) {
  *noiseCount = thrust::count(thrust::device, d_cluster, d_cluster + DATASET_COUNT, NOISE);
  int *d_localCluster;
  gpuErrchk(cudaMalloc((void **)&d_localCluster, sizeof(int) * DATASET_COUNT));
  thrust::copy(thrust::device, d_cluster, d_cluster + DATASET_COUNT, d_localCluster);
  thrust::sort(thrust::device, d_localCluster, d_localCluster + DATASET_COUNT);
  *clusterCount = thrust::unique(thrust::device, d_localCluster, d_localCluster + DATASET_COUNT) - d_localCluster - 1;
  cudaFree(d_localCluster);
}

void TestGetDbscanResult(int *d_cluster, int *runningCluster, int *clusterCount,
                         int *noiseCount) {
  *noiseCount = thrust::count(thrust::device, d_cluster, d_cluster + DATASET_COUNT, NOISE);
  int *d_localCluster;
  gpuErrchk(cudaMalloc((void **)&d_localCluster, sizeof(int) * DATASET_COUNT));
  thrust::copy(thrust::device, d_cluster, d_cluster + DATASET_COUNT, d_localCluster);
  thrust::sort(thrust::device, d_localCluster, d_localCluster + DATASET_COUNT);
  *clusterCount = thrust::unique(thrust::device, d_localCluster, d_localCluster + DATASET_COUNT) - d_localCluster - 1;
  cudaFree(d_localCluster);

}

__global__ void DBSCAN_ONE_INSTANCE(double *dataset, int *cluster,
                                    int *seedList, int *seedLength,
                                    int *collisionMatrix, int *extraCollision,
                                    int *results,
                                    struct IndexStructure **indexBuckets,
                                    int *indexesStack, int *dataValue,
                                    double *upperBounds, double *binWidth) {
  // Point ID to expand by a block
  __shared__ int pointID;

  // Neighbors to store of neighbors points exceeds minpoints
  __shared__ int neighborBuffer[MINPTS];

  // It counts the total neighbors
  __shared__ int neighborCount;

  // ChainID is basically blockID
  __shared__ int chainID;

  // Store the point from pointID
  __shared__ double point[DIMENSION];

  // Length of the seedlist to check its size
  __shared__ int currentSeedLength;

  __shared__ int resultId;


  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int x = threadId; x < THREAD_BLOCKS * THREAD_BLOCKS;
       x = x + THREAD_BLOCKS * THREAD_COUNT) {
    collisionMatrix[x] = UNPROCESSED;
  }
  for (int x = threadId; x < THREAD_BLOCKS * EXTRA_COLLISION_SIZE;
       x = x + THREAD_BLOCKS * THREAD_COUNT) {
    extraCollision[x] = UNPROCESSED;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
  }
  __syncthreads();

  while (seedLength[chainID] != 0) {
    for (int x = threadId; x < THREAD_BLOCKS * POINTS_SEARCHED;
         x = x + THREAD_BLOCKS * THREAD_COUNT) {
      results[x] = UNPROCESSED;
    }
    __syncthreads();

    // Assign chainID, current seed length and pointID
    if (threadIdx.x == 0) {
      currentSeedLength = seedLength[chainID];
      pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
    }
    __syncthreads();

    // Check if the point is already processed
    if (threadIdx.x == 0) {
      seedLength[chainID] = currentSeedLength - 1;
      neighborCount = 0;
      for (int x = 0; x < DIMENSION; x++) {
        point[x] = dataset[pointID * DIMENSION + x];
      }
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////

    searchPoints(point, chainID, dataset, results, indexBuckets, indexesStack,
                 dataValue, upperBounds, binWidth);

    __syncthreads();

    for (int k = 0; k < POINTS_SEARCHED; k++) {
      if (threadIdx.x == 0) {
        resultId = results[chainID * POINTS_SEARCHED + k];
      }
      __syncthreads();

      if (resultId == -1) break;

      for (int i = threadIdx.x + indexBuckets[resultId]->dataBegin;
           i < indexBuckets[resultId]->dataEnd; i = i + THREAD_COUNT) {
        register double comparingPoint[DIMENSION];

        for (int x = 0; x < DIMENSION; x++) {
          comparingPoint[x] = dataset[dataValue[i] * DIMENSION + x];
        }

        register double distance = 0;
        for (int x = 0; x < DIMENSION; x++) {
          distance +=
              (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
        }

        if (distance <= EPS * EPS) {
          register int currentNeighborCount = atomicAdd(&neighborCount, 1);
          if (currentNeighborCount >= MINPTS) {
            MarkAsCandidate(dataValue[i], chainID, cluster, seedList,
                            seedLength, collisionMatrix, extraCollision);
          } else {
            neighborBuffer[currentNeighborCount] = dataValue[i];
          }
        }
      }
      __syncthreads();
    }
    __syncthreads();

    ///////////////////////////////////////////////////////////////////////////////////

    if (neighborCount >= MINPTS) {
      cluster[pointID] = chainID;
      for (int i = threadIdx.x; i < MINPTS; i = i + THREAD_COUNT) {
        MarkAsCandidate(neighborBuffer[i], chainID, cluster, seedList,
                        seedLength, collisionMatrix, extraCollision);
      }
    } else {
      cluster[pointID] = NOISE;
    }

    __syncthreads();
    ///////////////////////////////////////////////////////////////////////////////////

    if (threadIdx.x == 0 && seedLength[chainID] >= MAX_SEEDS) {
      seedLength[chainID] = MAX_SEEDS - 1;
    }
    __syncthreads();
  }
}



__global__ void COLLISION_DETECTION(int *collisionMatrix, int *extraCollision,
                                    int *cluster, int *clusterMap, int*clusterCountMap, int* runningCluster) {
  if (threadIdx.x == 0) {
    clusterMap[blockIdx.x] = blockIdx.x;
    clusterCountMap[blockIdx.x] = UNPROCESSED;
  }
  __syncthreads();

  if (blockIdx.x == 0) {

    __shared__ int blockSet[THREAD_BLOCKS];
    __shared__ int blocksetCount;
    __shared__ int curBlock;
    __shared__ int expansionQueue[THREAD_BLOCKS];
    __shared__ int finalQueue[THREAD_BLOCKS];
    __shared__ int expansionQueueCount;
    __shared__ int finalQueueCount;
    __shared__ int expandBlock;
    

    for (int i = threadIdx.x; i < THREAD_BLOCKS; i = i + THREAD_COUNT) {
      blockSet[i] = i;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      blocksetCount = THREAD_BLOCKS;
    }
    __syncthreads();

    while (blocksetCount > 0) {
      if (threadIdx.x == 0) {
        curBlock = blockSet[0];
        expansionQueueCount = 0;
        finalQueueCount = 0;
        expansionQueue[expansionQueueCount++] = curBlock;
        finalQueue[finalQueueCount++] = curBlock;
      }
      __syncthreads();

      while (expansionQueueCount > 0) {
        if (threadIdx.x == 0) {
          expandBlock = expansionQueue[--expansionQueueCount];

          thrust::remove(thrust::device, expansionQueue,
                         expansionQueue + THREAD_BLOCKS, expandBlock);
          thrust::remove(thrust::device, blockSet, blockSet + THREAD_BLOCKS,
                         expandBlock);
          blocksetCount--;
        }
        __syncthreads();

        for (int x = threadIdx.x; x < THREAD_BLOCKS; x = x + THREAD_COUNT) {
          if (x != expandBlock) {
            if (collisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 &&
                thrust::find(thrust::device, blockSet, blockSet + blocksetCount,
                             x) != blockSet + blocksetCount) {
              if (thrust::find(thrust::device, expansionQueue,
                               expansionQueue + expansionQueueCount,
                               x) == expansionQueue + expansionQueueCount) {
                int oldExpansionQueueCount = atomicAdd(&expansionQueueCount, 1);
                expansionQueue[oldExpansionQueueCount] = x;
              }

              if (thrust::find(thrust::device, finalQueue,
                               finalQueue + finalQueueCount,
                               x) == finalQueue + finalQueueCount) {
                int oldFinalQueueCount = atomicAdd(&finalQueueCount, 1);
                finalQueue[oldFinalQueueCount] = x;
              }
            }
          }

          __syncthreads();
        }
      };
      __syncthreads();

      for (int c = threadIdx.x; c < finalQueueCount; c = c + THREAD_COUNT) {
        clusterMap[finalQueue[c]] = curBlock;
      }
      __syncthreads();
    };

    if(threadIdx.x == 0) {
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (clusterCountMap[clusterMap[x]] == UNPROCESSED) {
          clusterCountMap[clusterMap[x]] = runningCluster[0]++;
        }
      }
    }
    __syncthreads();

  }
  __syncthreads();
  
}

__global__ void COLLISION_MERGE(int *collisionMatrix, int *extraCollision,
  int *cluster,  int *clusterMap, int*clusterCountMap) {

  __shared__ int chainID;

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
  }
  __syncthreads();
  

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < DATASET_COUNT; i = i + THREAD_BLOCKS * THREAD_COUNT) {
    if (cluster[i] >= 0 && cluster[i] < THREAD_BLOCKS) {
      cluster[i] = clusterCountMap[clusterMap[cluster[i]]];
    }
  }
  __syncthreads();

  for (int y = 0; y < EXTRA_COLLISION_SIZE; y++) {
    if (extraCollision[chainID * EXTRA_COLLISION_SIZE + y] != UNPROCESSED) {
      for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
        if (extraCollision[chainID * EXTRA_COLLISION_SIZE + y] == cluster[i])
          cluster[i] = extraCollision[chainID * EXTRA_COLLISION_SIZE];
        if (y == 0 && cluster[i] == clusterCountMap[clusterMap[chainID]])
          cluster[i] = extraCollision[chainID * EXTRA_COLLISION_SIZE];
      }
    }
  }
  __syncthreads();
}


bool TestMonitorSeedPoints(vector<int> &unprocessedPoints,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_results, int *d_clusterMap, int *d_clusterCountMap, int* d_runningCluster) {
  int *localSeedLength;
  localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localSeedLength, d_seedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int *localSeedList;
  localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));


  int completeSeedListFirst = false;
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    if (localSeedLength[i] > 0) {
      completeSeedListFirst = true;
    }
  }
  if (completeSeedListFirst) {
    free(localSeedList);
    free(localSeedLength);
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  gpuErrchk(cudaDeviceSynchronize());
  COLLISION_DETECTION<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(d_collisionMatrix, d_extraCollision,
    d_cluster, d_clusterMap, d_clusterCountMap, d_runningCluster);
  gpuErrchk(cudaDeviceSynchronize());

  COLLISION_MERGE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(d_collisionMatrix, d_extraCollision,
    d_cluster, d_clusterMap, d_clusterCountMap);
  gpuErrchk(cudaDeviceSynchronize());

  //////////////////////////////////////////////////////////////////////////////////////////
  int* d_datasetSequence;
  gpuErrchk(cudaMalloc((void **)&d_datasetSequence, sizeof(int) * DATASET_COUNT));
  int *d_tempCluster;
  gpuErrchk(cudaMalloc((void **)&d_tempCluster, sizeof(int) * DATASET_COUNT));

  thrust::sequence(thrust::device, d_datasetSequence, d_datasetSequence + DATASET_COUNT);
  thrust::copy(thrust::device, d_cluster, d_cluster + DATASET_COUNT, d_tempCluster);
  thrust::sort_by_key(thrust::device, d_tempCluster, d_tempCluster + DATASET_COUNT, d_datasetSequence);

  thrust::pair<int *, int *> dataPositioned;
  dataPositioned = thrust::equal_range(thrust::device, d_tempCluster, d_tempCluster + DATASET_COUNT, UNPROCESSED);
  int first = dataPositioned.first - d_tempCluster;
  int last = dataPositioned.second  - d_tempCluster;
  
  int* datasetSequence = (int*)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(datasetSequence, d_datasetSequence, sizeof(int) * DATASET_COUNT, cudaMemcpyDeviceToHost));
  
  int blockCount = 0;
  for(int x = first; x < last; x++) {
    if(blockCount > THREAD_BLOCKS) break;
    localSeedList[blockCount * MAX_SEEDS] = datasetSequence[x];
    localSeedLength[blockCount] = 1;
    blockCount++;
  }

  cudaFree(d_datasetSequence);
  cudaFree(d_tempCluster);
  free(datasetSequence);

  //////////////////////////////////////////////////////////////////////////////////////////

  // Finally, transfer back the CPU memory to GPU and run DBSCAN process

  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  // Free CPU memories
  free(localSeedList);
  free(localSeedLength);

  if(first == last) {
    return true;
  }

  return false;
}