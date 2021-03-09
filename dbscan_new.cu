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

using namespace std;

// Number of data in dataset to use
// #define DATASET_COUNT 1864620
#define DATASET_COUNT 50000000

// Dimension of the dataset
#define DIMENSION 2

// Maximum size of seed list
#define MAX_SEEDS 256

// Extra collission size to detect final clusters collision
#define EXTRA_COLLISION_SIZE 512

// Number of blocks
#define THREAD_BLOCKS 2048

// Number of threads per block
#define THREAD_COUNT 1024

// Status of points that are not clusterized
#define UNPROCESSED -1

// Status for noise point
#define NOISE -2

// Minimum number of points in DBSCAN
#define MINPTS 4

#define TREE_LEVELS (DIMENSION + 1)

// Epslion value in DBSCAN
#define EPS 0.01

#define RANGE 2

#define PARTITION_SIZE 1000

#define POINTS_SEARCHED 9

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* GPU ERROR function checks for potential erros in cuda function execution
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* INDEXING datastructure and functions
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/

struct __align__(8) IndexStructure {
  int id;
  int dimension;
  int dataBegin;
  int dataEnd;
  double range[RANGE];
  int childBuckets[PARTITION_SIZE];
};

__global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                                   double *minPoints, double *binWidth,
                                   int *results,
                                   struct IndexStructure **indexBuckets,
                                   int *dataKey, int *dataValue);

__global__ void INDEXING_ADJUSTMENT(int *indexTreeMetaData,
                                    struct IndexStructure **indexBuckets,
                                    int *dataKey);

__device__ void indexConstruction(int dimension, int *indexTreeMetaData,
                                  double *minPoints, double *binWidth,
                                  struct IndexStructure **indexBuckets);

__device__ void insertData(int id, double *dataset,
                           struct IndexStructure **indexBuckets, int *dataKey,
                           int *dataValue);

__device__ void searchPoints(double *data, int chainID, double *dataset,
                             int *results, struct IndexStructure **indexBuckets,

                             int *indexesStack, int *dataValue);

int compare(const void *a, const void *b) { return (*(int *)a - *(int *)b); }

int compareDouble(const void *a, const void *b) {
  return (*(double *)a - *(double *)b);
}

__global__ void COLLISION_DETECTION(int *collisionMatrix, int *extraCollision,
  int *cluster, int *seedList, int *seedLength,
  int *runningCluster);
/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Declare CPU and GPU Functions
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int ImportDataset(char const *fname, double *dataset);

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *d_cluster, int *d_seedList, int *d_seedLength, int *d_results);

void GetDbscanResult(double *d_dataset, int *d_cluster, int *runningCluster,
                     int *clusterCount, int *noiseCount);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,

                       int *indexesStack, int *dataValue);

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *collisionMatrix, int *extraCollision);
/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Main CPU function
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int main(int argc, char **argv) {
  char inputFname[500];
  if (argc != 2) {
    fprintf(stderr, "Please provide the dataset file path in the arguments\n");
    exit(0);
  }

  // Get the dataset file name from argument
  strcpy(inputFname, argv[1]);
  printf("Using dataset file %s\n", inputFname);

  double *importedDataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);

  // Import data from dataset
  int ret = ImportDataset(inputFname, importedDataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  // Check if the data parsed is correct
  for (int i = 0; i < DIMENSION; i++) {
    printf("Sample Data %f\n", importedDataset[i]);
  }

  // Get the total count of dataset
  vector<int> unprocessedPoints;
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }

  printf("Preprocessed %lu data in dataset\n", unprocessedPoints.size());

  // Reset the GPU device for potential memory issues
  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  // Start the time
  clock_t totalTimeStart, totalTimeStop, indexingStart, indexingStop;
  float totalTime = 0.0;
  float indexingTime = 0.0;
  totalTimeStart = clock();

  /**
 **************************************************************************
 * CUDA Memory allocation
 **************************************************************************
 */
  double *d_dataset;
  int *d_cluster;
  int *d_seedList;
  int *d_seedLength;
  int *d_collisionMatrix;
  int *d_extraCollision;

  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));

  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMalloc((void **)&d_seedLength, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  /**
 **************************************************************************
 * Indexing Memory allocation
 **************************************************************************
 */

  indexingStart = clock();

  int *d_indexTreeMetaData;
  int *d_results;
  double *d_minPoints;
  double *d_binWidth;

  gpuErrchk(cudaMalloc((void **)&d_indexTreeMetaData,
                       sizeof(int) * TREE_LEVELS * RANGE));

  gpuErrchk(cudaMalloc((void **)&d_results,
                       sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  gpuErrchk(cudaMalloc((void **)&d_minPoints, sizeof(double) * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_binWidth, sizeof(double) * DIMENSION));

  gpuErrchk(cudaMemset(d_results, UNPROCESSED,
                       sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  /**
 **************************************************************************
 * Assignment with default values
 **************************************************************************
 */
  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_cluster, UNPROCESSED, sizeof(int) * DATASET_COUNT));

  gpuErrchk(
      cudaMemset(d_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMemset(d_seedLength, 0, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  /**
**************************************************************************
* Initialize index structure
**************************************************************************
*/
  double maxPoints[DIMENSION];
  double minPoints[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    maxPoints[j] = 0;
    minPoints[j] = 999999999;
  }

  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      if (importedDataset[i * DIMENSION + j] > maxPoints[j]) {
        maxPoints[j] = importedDataset[i * DIMENSION + j];
      }
      if (importedDataset[i * DIMENSION + j] < minPoints[j]) {
        minPoints[j] = importedDataset[i * DIMENSION + j];
      }
    }
  }

  for (int i = 0; i < DIMENSION; i++) {
    printf("Level %d Max: %f\n", i, maxPoints[i]);
    printf("Level %d Min: %f\n", i, minPoints[i]);
  }

  double binWidth[DIMENSION];
  double minBinSize = 99999999;
  for (int x = 0; x < DIMENSION; x++) {
    binWidth[x] = (double)(maxPoints[x] - minPoints[x]) / PARTITION_SIZE;
    if (minBinSize >= binWidth[x]) {
      minBinSize = binWidth[x];
    }
  }
  for (int x = 0; x < DIMENSION; x++) {
    printf("#%d Bin Width: %lf\n", x, binWidth[x]);
  }
  printf("Min Bin Size: %lf\n", minBinSize);

  if (minBinSize < EPS) {
    printf("Bin width (%f) is less than EPS(%f).", minBinSize, EPS);
    exit(0);
  }

  // Level Partition
  int treeLevelPartition[TREE_LEVELS] = {1};

  for (int i = 0; i < DIMENSION; i++) {
    treeLevelPartition[i + 1] = PARTITION_SIZE;
  }

  int childItems[TREE_LEVELS];
  int startEndIndexes[TREE_LEVELS * RANGE];

  int mulx = 1;
  for (int k = 0; k < TREE_LEVELS; k++) {
    mulx *= treeLevelPartition[k];
    childItems[k] = mulx;
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    if (i == 0) {
      startEndIndexes[i * RANGE + 0] = 0;
      startEndIndexes[i * RANGE + 1] = 1;
      continue;
    }
    startEndIndexes[i * RANGE + 0] = startEndIndexes[((i - 1) * RANGE) + 1];
    startEndIndexes[i * RANGE + 1] = startEndIndexes[i * RANGE + 0];
    for (int k = 0; k < childItems[i - 1]; k++) {
      startEndIndexes[i * RANGE + 1] += treeLevelPartition[i];
    }
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    printf("#%d ", i);
    printf("Partition: %d ", treeLevelPartition[i]);
    printf("Range: %d %d\n", startEndIndexes[i * RANGE + 0],
           startEndIndexes[i * RANGE + 1]);
  }

  gpuErrchk(cudaMemcpy(d_minPoints, minPoints, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_binWidth, binWidth, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_indexTreeMetaData, startEndIndexes,
                       sizeof(int) * TREE_LEVELS * RANGE,
                       cudaMemcpyHostToDevice));

  int indexedStructureSize = startEndIndexes[DIMENSION * RANGE + 1];

  printf("Index Structure Size: %lf GB.\n",
         (sizeof(struct IndexStructure) * indexedStructureSize) /
             (1024 * 1024 * 1024.0));

  // Allocate memory for index buckets
  struct IndexStructure **d_indexBuckets, *d_currentIndexBucket;

  gpuErrchk(cudaMalloc((void **)&d_indexBuckets,
                       sizeof(struct IndexStructure *) * indexedStructureSize));

  for (int i = 0; i < indexedStructureSize; i++) {
    gpuErrchk(cudaMalloc((void **)&d_currentIndexBucket,
                         sizeof(struct IndexStructure)));
    gpuErrchk(cudaMemcpy(&d_indexBuckets[i], &d_currentIndexBucket,
                         sizeof(struct IndexStructure *),
                         cudaMemcpyHostToDevice));
  }

  // Allocate memory for current indexes stack
  int indexBucketSize = 1;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }

  indexBucketSize = indexBucketSize * THREAD_BLOCKS;

  int *d_indexesStack;

  gpuErrchk(
      cudaMalloc((void **)&d_indexesStack, sizeof(int) * indexBucketSize));

  cudaFree(d_currentIndexBucket);

  /**
 **************************************************************************
 * Data key-value pair
 **************************************************************************
 */
  int *d_dataKey;
  int *d_dataValue;

  gpuErrchk(cudaMalloc((void **)&d_dataKey, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_dataValue, sizeof(int) * DATASET_COUNT));

  /**
 **************************************************************************
 * Start Indexing first
 **************************************************************************
 */
  gpuErrchk(cudaDeviceSynchronize());

  INDEXING_STRUCTURE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_dataset, d_indexTreeMetaData, d_minPoints, d_binWidth, d_results,
      d_indexBuckets, d_dataKey, d_dataValue);
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(d_indexTreeMetaData);
  cudaFree(d_minPoints);

  /**
 **************************************************************************
 * Sorting and adjusting Data key-value pair
 **************************************************************************
 */

  thrust::sort_by_key(thrust::device, d_dataKey, d_dataKey + DATASET_COUNT,
                      d_dataValue);

  gpuErrchk(cudaDeviceSynchronize());

  INDEXING_ADJUSTMENT<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_indexTreeMetaData, d_indexBuckets, d_dataKey);

  gpuErrchk(cudaDeviceSynchronize());

  indexingStop = clock();

  printf("Index structure created.\n");

  /**
 **************************************************************************
 * Start the DBSCAN algorithm
 **************************************************************************
 */

  // Keep track of number of cluster formed without global merge
  int* runningCluster = (int*)malloc(sizeof(int));
  runningCluster[0] = 0;

  int *d_runningCluster;
  gpuErrchk(cudaMalloc((void **)&d_runningCluster, sizeof(int)));
  gpuErrchk(cudaMemset(d_runningCluster, 0, sizeof(int)));

  // Global cluster count
  int clusterCount = 0;

  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to conmtrol the while loop
  bool exit = false;

  while (!exit) {
    // Monitor the seed list and return the comptetion status of points
    int completed = MonitorSeedPoints(
        unprocessedPoints, d_cluster, d_seedList, d_seedLength, d_results);
    
    gpuErrchk(cudaMemcpy(runningCluster, d_runningCluster, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Running cluster %d, unprocessed points: %lu\n", runningCluster[0],
           unprocessedPoints.size());

    // If all points are processed, exit
    if (completed) {
      exit = true;
    }

    if (exit) break;

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_extraCollision, d_results, d_indexBuckets, d_indexesStack,
        d_dataValue);
    gpuErrchk(cudaDeviceSynchronize());
    COLLISION_DETECTION<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(d_collisionMatrix, d_extraCollision,
      d_cluster, d_seedList, d_seedLength,
      d_runningCluster);
    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
 **************************************************************************
 * End DBSCAN and show the results
 **************************************************************************
 */

  // Get the DBSCAN result
  GetDbscanResult(d_dataset, d_cluster, runningCluster, &clusterCount,
                  &noiseCount);

  totalTimeStop = clock();
  totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
  indexingTime = (float)(indexingStop - indexingStart) / CLOCKS_PER_SEC;

  printf("==============================================\n");
  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);
  printf("==============================================\n");
  printf("Indexing Time: %3.2f seconds\n", indexingTime);
  printf("Total Time: %3.2f seconds\n", totalTime);
  printf("==============================================\n");

  /**
 **************************************************************************
 * Free CUDA memory allocations
 **************************************************************************
 */

  cudaFree(d_dataset);
  cudaFree(d_cluster);
  cudaFree(d_seedList);
  cudaFree(d_seedLength);
  cudaFree(d_collisionMatrix);
  cudaFree(d_extraCollision);

  cudaFree(d_results);
  cudaFree(d_indexBuckets);
  cudaFree(d_indexesStack);

  cudaFree(d_dataKey);
  cudaFree(d_dataValue);
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Monitor Seed Points performs the following operations.
* 1) Check if the seed list is empty. If it is empty check the refill seed list
* else, return false to process next seed point by DBSCAN.
* 2) If seed list is empty, It will check refill seed list and fill the points
* from refill seed list to seed list
* 3) If seed list and refill seed list both are empty, then check for the
* collision matrix and form a cluster by merging chains.
* 4) After clusters are merged, new points are assigned to seed list
* 5) Lastly, It checks if all the points are processed. If so it will return
* true and DBSCAN algorithm will exit.
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *d_cluster, int *d_seedList, int *d_seedLength, int *d_results) {
  /**
 **************************************************************************
 * Copy GPU variables content to CPU variables for seed list management
 **************************************************************************
 */
  int *localSeedLength;
  localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localSeedLength, d_seedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

  int *localSeedList;
  localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
  gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyDeviceToHost));

  gpuErrchk(
      cudaMemset(d_results, UNPROCESSED, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  /**
 **************************************************************************
 * After finilazing the cluster, check the remaining points and
 * insert one point to each of the seedlist
 **************************************************************************
 */

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

  /**
**************************************************************************
* FInally, transfer back the CPU memory to GPU and run DBSCAN process
**************************************************************************
*/

  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  /**
 **************************************************************************
 * Free CPU memory allocations
 **************************************************************************
 */

  free(localCluster);
  free(localSeedList);
  free(localSeedLength);

  if (complete == THREAD_BLOCKS) {
    return true;
  }

  return false;
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Get DBSCAN result
* Get the final cluster and print the overall result
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
void GetDbscanResult(double *d_dataset, int *d_cluster, int *runningCluster,
                     int *clusterCount, int *noiseCount) {
  /**
**************************************************************************
* Print the cluster and noise results
**************************************************************************
*/

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  double *dataset;
  dataset = (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);
  gpuErrchk(cudaMemcpy(dataset, d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyDeviceToHost));

  map<int, int> finalClusterMap;
  int localClusterCount = 0;
  int localNoiseCount = 0;
  for (int i = THREAD_BLOCKS; i <= (runningCluster[0]) + THREAD_BLOCKS; i++) {
    bool found = false;
    for (int j = 0; j < DATASET_COUNT; j++) {
      if (localCluster[j] == i) {
        found = true;
        break;
      }
    }
    if (found) {
      ++localClusterCount;
      finalClusterMap[i] = localClusterCount;
    }
  }
  for (int j = 0; j < DATASET_COUNT; j++) {
    if (localCluster[j] == NOISE) {
      localNoiseCount++;
    }
  }

  *clusterCount = localClusterCount;
  *noiseCount = localNoiseCount;

  for (int j = 0; j < DATASET_COUNT; j++) {
    if (finalClusterMap[localCluster[j]] >= 0) {
      localCluster[j] = finalClusterMap[localCluster[j]];
    } else {
      localCluster[j] = 0;
    }
  }

  free(localCluster);
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* DBSCAN: Main kernel function of the algorithm
* It does the following functions.
* 1) Every block gets a point from seedlist to expand. If these points are
* processed already, it returns
* 2) It expands the points by finding neighbors points
* 3) Checks for the collision and mark the collision in collision matrix
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,

                       int *indexesStack, int *dataValue) {
  /**
 **************************************************************************
 * Define shared variables
 **************************************************************************
 */

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

  __shared__ int resultId;

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
  }
  __syncthreads();

  // Complete the seedlist to proceed.

  while (seedLength[chainID] > 0) {
    /**
**************************************************************************
* Get current chain length, and If its zero, exit
**************************************************************************
*/

    // Check if the point is already processed
    if (threadIdx.x == 0) {
      pointID = seedList[chainID * MAX_SEEDS + seedLength[chainID] - 1];
      seedLength[chainID] = seedLength[chainID] - 1;
      neighborCount = 0;
      for (int x = 0; x < DIMENSION; x++) {
        point[x] = dataset[pointID * DIMENSION + x];
      }
    }
    __syncthreads();

    /**
**************************************************************************
* Find the neighbors of the pointID
* Mark point as candidate if points are more than min points
* Keep record of left over neighbors in neighborBuffer
**************************************************************************
*/
searchPoints(point, chainID, dataset, results, indexBuckets, indexesStack,
                 dataValue);

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


    /**
**************************************************************************
* Mark the left over neighbors in neighborBuffer as cluster member
* If neighbors are less than MINPTS, assign pointID with noise
**************************************************************************
*/

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

    /**
**************************************************************************
* Check Thread length, If it exceeds MAX limit the length
* As seedlist wont have data beyond its max length
**************************************************************************
*/

    if (threadIdx.x == 0 && seedLength[chainID] >= MAX_SEEDS) {
      seedLength[chainID] = MAX_SEEDS - 1;
    }
    __syncthreads();
  }
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Mark as candidate
* It does the following functions:
* 1) Mark the neighbor's cluster with chainID if its old state is unprocessed
* 2) If the oldstate is unprocessed, insert the neighnor point to seed list
* 3) if the seed list exceeds max value, insert into refill seed list
* 4) If the old state is less than THREAD BLOCK, record the collision in
* collision matrix
* 5) If the old state is greater than THREAD BLOCK, record the collision
* in extra collision
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,
                                int *collisionMatrix, int *extraCollision) {
  /**
**************************************************************************
* Get the old cluster state of the neighbor
* If the state is unprocessed, assign it with chainID
**************************************************************************
*/
  register int oldState =
      atomicCAS(&(cluster[neighborID]), UNPROCESSED, chainID);

  /**
 **************************************************************************
 * For unprocessed old state of neighbors, add them to seedlist and
 * refill seedlist
 **************************************************************************
 */
  if (oldState == UNPROCESSED) {
    register int sl = atomicAdd(&(seedLength[chainID]), 1);
    if (sl < MAX_SEEDS) {
      seedList[chainID * MAX_SEEDS + sl] = neighborID;
    }
  }

  /**
 **************************************************************************
 * If the old state is greater than thread block, record the extra collisions
 **************************************************************************
 */

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

  /**
 **************************************************************************
 * If the old state of neighbor is not noise, not member of chain and cluster
 * is within THREADBLOCK, maek the collision between old and new state
 **************************************************************************
 */
  else if (oldState != NOISE && oldState != chainID &&
           oldState < THREAD_BLOCKS) {
    collisionMatrix[oldState * THREAD_BLOCKS + chainID] = 1;
    collisionMatrix[chainID * THREAD_BLOCKS + oldState] = 1;
  }

  /**
 **************************************************************************
 * If the old state is noise, assign it to chainID cluster
 **************************************************************************
 */
  else if (oldState == NOISE) {
    oldState = atomicCAS(&(cluster[neighborID]), NOISE, chainID);
  }
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Helper functions for index construction and points search...
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/

__global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                                   double *minPoints, double *binWidth,
                                   int *results,
                                   struct IndexStructure **indexBuckets,
                                   int *dataKey, int *dataValue) {
  if(blockIdx.x < DIMENSION) {
    indexConstruction(blockIdx.x, indexTreeMetaData, minPoints, binWidth, indexBuckets);
  }
  __syncthreads();

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < DATASET_COUNT;
       i = i + THREAD_COUNT * THREAD_BLOCKS) {
    insertData(i, dataset, indexBuckets, dataKey, dataValue);
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

    dataPositioned = thrust::equal_range(thrust::device, dataKey,
                                         dataKey + DATASET_COUNT, idx);

    indexBuckets[idx]->dataBegin = dataPositioned.first - dataKey;
    indexBuckets[idx]->dataEnd = dataPositioned.second - dataKey;
  }
  __syncthreads();
}

__device__ void indexConstruction(int level, int *indexTreeMetaData,
                                  double *minPoints, double *binWidth,
                                  struct IndexStructure **indexBuckets) {
  
  
  for (int k = threadIdx.x + indexTreeMetaData[level * RANGE + 0];
       k < indexTreeMetaData[level * RANGE + 1]; k = k + THREAD_COUNT) {

    for (int i = 0; i < PARTITION_SIZE; i++) {
      
      int currentBucketIndex =
          indexTreeMetaData[level * RANGE + 1] + i +
          (k - indexTreeMetaData[level * RANGE + 0]) * PARTITION_SIZE;

      indexBuckets[k]->dimension = level;
      indexBuckets[currentBucketIndex]->dimension = level + 1;

      indexBuckets[currentBucketIndex]->id = currentBucketIndex;
      indexBuckets[k]->childBuckets[i] = currentBucketIndex;

      double leftPoint = minPoints[level] + i * binWidth[level];
      double rightPoint = leftPoint + binWidth[level];

      if (i == 0) leftPoint = leftPoint - binWidth[level];
      if (i == PARTITION_SIZE - 1) rightPoint = rightPoint + binWidth[level];

      indexBuckets[currentBucketIndex]->range[0] = leftPoint;
      indexBuckets[currentBucketIndex]->range[1] = rightPoint;

    }
  }
  __syncthreads();
}

__device__ void insertData(int id, double *dataset,
                           struct IndexStructure **indexBuckets, int *dataKey,
                           int *dataValue) {
  double data[DIMENSION];
  for (int j = 0; j < DIMENSION; j++) {
    data[j] = dataset[id * DIMENSION + j];
  }

  int currentIndex = 0;
  bool found = false;

  while (!found) {
    if(indexBuckets[currentIndex]->dimension >= DIMENSION) break;
    for (int k = 0; k < PARTITION_SIZE; k++) {
      double comparingData = data[indexBuckets[currentIndex]->dimension];
      double leftRange =
          indexBuckets[indexBuckets[currentIndex]->childBuckets[k]]->range[0];
      double rightRange =
          indexBuckets[indexBuckets[currentIndex]->childBuckets[k]]->range[1];

      if (comparingData >= leftRange && comparingData < rightRange) {
        if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {
          dataValue[id] = id;
          dataKey[id] = indexBuckets[currentIndex]->childBuckets[k];
          found = true;
          break;
        }
        currentIndex = indexBuckets[currentIndex]->childBuckets[k];
        break;
      }
    }
  }
}

__device__ void searchPoints(double *data, int chainID, double *dataset,
                             int *results, struct IndexStructure **indexBuckets,
                             int *indexesStack, int *dataValue) {
  __shared__ int resultsCount;
  __shared__ int indexBucketSize;
  __shared__ int currentIndex;
  __shared__ int currentIndexSize;
  __shared__ int stopTraverse;
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
    stopTraverse = 0;
  }
  __syncthreads();

  while (currentIndexSize > indexBucketSize) {
    if (threadIdx.x == 0) {
      currentIndexSize = currentIndexSize - 1;
      currentIndex = indexesStack[currentIndexSize];
      stopTraverse = 0;
      comparingData = data[indexBuckets[currentIndex]->dimension];
    }
    __syncthreads();

    for (int k = threadIdx.x; k < PARTITION_SIZE; k = k + THREAD_COUNT) {
      if (stopTraverse == 1) break;
      register double leftRange =
          indexBuckets[indexBuckets[currentIndex]->childBuckets[k]]->range[0];
      register double rightRange =
          indexBuckets[indexBuckets[currentIndex]->childBuckets[k]]->range[1];
      if (comparingData >= leftRange && comparingData < rightRange) {
        if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {



          int oldResultsCount = atomicAdd(&resultsCount, 1);
          results[chainID * POINTS_SEARCHED + oldResultsCount] =
              indexBuckets[currentIndex]->childBuckets[k];

          if(k > 0) {
            oldResultsCount = atomicAdd(&resultsCount, 1);
            results[chainID * POINTS_SEARCHED + oldResultsCount] =
              indexBuckets[currentIndex]->childBuckets[k] - 1;
          }

          if(k < PARTITION_SIZE - 1) {
            oldResultsCount = atomicAdd(&resultsCount, 1);
            results[chainID * POINTS_SEARCHED + oldResultsCount] =
              indexBuckets[currentIndex]->childBuckets[k] + 1;
          }
            
          atomicCAS(&stopTraverse, 0, 1);
          break;
        }

        int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
        indexesStack[oldCurrentIndexSize] =
            indexBuckets[currentIndex]->childBuckets[k];
        if (k > 0) {
          int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
          indexesStack[oldCurrentIndexSize] =
              indexBuckets[currentIndex]->childBuckets[k - 1];
        }
        if (k < PARTITION_SIZE - 1) {
          int oldCurrentIndexSize = atomicAdd(&currentIndexSize, 1);
          indexesStack[oldCurrentIndexSize] =
              indexBuckets[currentIndex]->childBuckets[k + 1];
        }
        atomicCAS(&stopTraverse, 0, 1);
      }
    }

    __syncthreads();
  }
}

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Import Dataset
* It imports the data from the file and store in dataset variable
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int ImportDataset(char const *fname, double *dataset) {
  FILE *fp = fopen(fname, "r");
  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }

  char buf[4096];
  unsigned long int cnt = 0;
  while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT * DIMENSION) {
    char *field = strtok(buf, ",");
    long double tmp;
    sscanf(field, "%Lf", &tmp);
    dataset[cnt] = tmp;
    cnt++;

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        long double tmp;
        sscanf(field, "%Lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;
      }
    }
  }
  fclose(fp);
  return 0;
}


__device__ int clusterMap[THREAD_BLOCKS];
__device__ int blockSet[THREAD_BLOCKS];
__device__ int blocksetCount;
__device__ int curBlock;
__device__ int expansionQueue[THREAD_BLOCKS];
__device__ int finalQueue[THREAD_BLOCKS];
__device__ int expansionQueueCount;
__device__ int finalQueueCount;
__device__ int expandBlock;

__global__ void COLLISION_DETECTION(int *collisionMatrix, int *extraCollision,
                                 int *cluster, int *seedList, int *seedLength,
                                 int *runningCluster) {
    
  
  
    

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < THREAD_BLOCKS; i++) {
        clusterMap[i] = i;
        blockSet[i] = i;
      }
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
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          int oldExpansionQueueCount = atomicSub(&expansionQueueCount, 1);
          expandBlock = expansionQueue[oldExpansionQueueCount - 1];
        }
        __syncthreads();

        thrust::remove(thrust::device, expansionQueue,
                       expansionQueue + THREAD_BLOCKS, expandBlock);
        thrust::remove(thrust::device, blockSet, blockSet + THREAD_BLOCKS,
                       expandBlock);

        if (threadIdx.x == 0 && blockIdx.x == 0) {
          atomicSub(&blocksetCount, 1);
        }
        __syncthreads();

        for (int x = blockIdx.x; x < THREAD_BLOCKS; x = x + THREAD_BLOCKS) {
          if (x == expandBlock) continue;
          if ((collisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 ||
               collisionMatrix[x * THREAD_BLOCKS + expandBlock]) &&
              thrust::find(thrust::device, blockSet, blockSet + THREAD_BLOCKS,
                           x) != blockSet + THREAD_BLOCKS) {
            if (thrust::find(thrust::device, expansionQueue,
                             expansionQueue + THREAD_BLOCKS,
                             x) == expansionQueue + THREAD_BLOCKS) {
              if(threadIdx.x == 0) {
                int oldExpansionQueueCount = atomicAdd(&expansionQueueCount, 1);
                expansionQueue[oldExpansionQueueCount] = x;
              }
              
            }

            if (thrust::find(thrust::device, finalQueue,
                             finalQueue + THREAD_BLOCKS,
                             x) == finalQueue + THREAD_BLOCKS) {
              if(threadIdx.x == 0) {
                int oldFinalQueueCount = atomicAdd(&finalQueueCount, 1);
                finalQueue[oldFinalQueueCount] = x;
              }
            }
            __syncthreads();
          }
          __syncthreads();
        }
      };
      if(blockIdx.x == 0) {
        for (int c = threadIdx.x; c < finalQueueCount; c = c + THREAD_COUNT) {
          clusterMap[finalQueue[c]] = curBlock;
        }
      }

      __syncthreads();
    };

    
    if(blockIdx.x == 0) {
      for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
        if (cluster[i] >= 0 && cluster[i] < THREAD_BLOCKS) {
          if (extraCollision[cluster[i] * EXTRA_COLLISION_SIZE] ==
              UNPROCESSED) {
            cluster[i] = clusterMap[cluster[i]];
          } else {
            cluster[i] = extraCollision[clusterMap[cluster[i]] *
                                                  EXTRA_COLLISION_SIZE];
          }
        }
        __syncthreads();
      }
  }
  __syncthreads();
    
    for (int i = blockIdx.x; i < THREAD_BLOCKS; i = i + THREAD_BLOCKS) {
      if (extraCollision[i * EXTRA_COLLISION_SIZE] != UNPROCESSED) continue;
      for (int x = threadIdx.x; x < DATASET_COUNT; x = x + THREAD_COUNT) {
        if (cluster[x] == i) {
          cluster[x] = runningCluster[0] + THREAD_BLOCKS;
        }
      }
      __syncthreads();
      if(threadIdx.x == 0) {
        atomicCAS(&extraCollision[i * EXTRA_COLLISION_SIZE], UNPROCESSED, runningCluster[0] + THREAD_BLOCKS);
        atomicAdd(&runningCluster[0], 1);
      }
      
      __syncthreads();
    }
    
    for (int i = blockIdx.x; i < THREAD_BLOCKS; i = i + THREAD_BLOCKS) {
      if (extraCollision[i * EXTRA_COLLISION_SIZE] == UNPROCESSED) continue;
      for (int x = threadIdx.x; x < DATASET_COUNT; x = x + THREAD_COUNT) {
        for (int j = 0; j < EXTRA_COLLISION_SIZE; j++) {
          if (extraCollision[i * EXTRA_COLLISION_SIZE + j] == UNPROCESSED)
            break;
          if (cluster[x] ==
              extraCollision[i * EXTRA_COLLISION_SIZE + j]) {
            cluster[x] =
                extraCollision[clusterMap[i] * EXTRA_COLLISION_SIZE];
          }
          __syncthreads();
        }
        __syncthreads();
      }
      __syncthreads();
    }
  
  if(blockIdx.x == 0) {
    for(int x = threadIdx.x; x < THREAD_BLOCKS*EXTRA_COLLISION_SIZE; x = x + THREAD_COUNT) {
      extraCollision[x] = UNPROCESSED;
    }
    __syncthreads();
    for(int x = threadIdx.x; x < THREAD_BLOCKS*THREAD_BLOCKS; x = x + THREAD_COUNT) {
      collisionMatrix[x] = UNPROCESSED;
    }
    __syncthreads();
  }
  
}