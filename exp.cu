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

#define RANGE 2
#define UNPROCESSED -1
#define NOISE -2

#define DIMENSION 2
#define TREE_LEVELS (DIMENSION + 1)

#define THREAD_BLOCKS 256
#define THREAD_COUNT 256

#define MAX_SEEDS 128
#define EXTRA_COLLISION_SIZE 512

__managed__ int MINPTS = 8;
__managed__ double EPS = 0.08;
__managed__ int DATASET_COUNT = 100000;

#define PARTITION_SIZE 100
#define POINTS_SEARCHED 9

#define PORTO 0
#define SPATIAL 0
#define NGSI 0
#define IONO2D 1

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

struct __align__(8) IndexStructure {
  int dimension;
  int dataBegin;
  int dataEnd;
  int childFrom;
};

using namespace std;

int ImportDataset(char const *fname, double *dataset);


bool MonitorSeedPoints(vector<int>& unprocessedPoints, int* runningCluster,
    int* d_cluster, int* d_seedList, int* d_seedLength,
    int* d_collisionMatrix, int* d_extraCollision,
    int* d_results, float* mergeTime);

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

             __global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                double *minPoints, double *binWidth,
                int *results,
                struct IndexStructure **indexBuckets,
                int *dataKey, int *dataValue,
                double *upperBounds);

__global__ void INDEXING_ADJUSTMENT(int *indexTreeMetaData,
                 struct IndexStructure **indexBuckets,
                 int *dataKey);

__device__ void indexConstruction(int dimension, int *indexTreeMetaData,
               double *minPoints, double *binWidth,
               struct IndexStructure **indexBuckets,
               double *upperBounds);

__device__ void insertData(int id, double *dataset,
        struct IndexStructure **indexBuckets, int *dataKey,
        int *dataValue, double *upperBounds,
        double *binWidth);

__device__ void searchPoints(double *data, int chainID, double *dataset,
          int *results, struct IndexStructure **indexBuckets,
          int *indexesStack, int *dataValue,
          double *upperBounds, double *binWidth);

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Main CPU function
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
void runDBSCAN(char* const filename, int datasetSize, double eps, int minPts) {

    EPS = eps;
    MINPTS = minPts;
    DATASET_COUNT = datasetSize;

  printf("Using dataset file %s\n", filename);

  double *importedDataset =
      (double *)malloc(sizeof(double) * DATASET_COUNT * DIMENSION);

  // Import data from dataset
  int ret = ImportDataset(filename, importedDataset);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return;
  }

  // Get the total count of dataset
  vector<int> unprocessedPoints;
  for (int x = 0; x < DATASET_COUNT; x++) {
    unprocessedPoints.push_back(x);
  }

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

  gpuErrchk(
      cudaMemset(d_results, -1, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

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

  double binWidth[DIMENSION];
  double minBinSize = 99999999;
  for (int x = 0; x < DIMENSION; x++) {
    binWidth[x] = (double)(maxPoints[x] - minPoints[x]) / PARTITION_SIZE;
    if (minBinSize >= binWidth[x]) {
      minBinSize = binWidth[x];
    }
  }

  if (minBinSize < EPS) {
    printf("Bin width (%f) is less than EPS(%f).\n", minBinSize, EPS);
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

  gpuErrchk(cudaMemcpy(d_minPoints, minPoints, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_binWidth, binWidth, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_indexTreeMetaData, startEndIndexes,
                       sizeof(int) * TREE_LEVELS * RANGE,
                       cudaMemcpyHostToDevice));

  int indexedStructureSize = startEndIndexes[DIMENSION * RANGE + 1];

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
  double *d_upperBounds;

  gpuErrchk(cudaMalloc((void **)&d_dataKey, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_dataValue, sizeof(int) * DATASET_COUNT));
  gpuErrchk(cudaMalloc((void **)&d_upperBounds,
                       sizeof(double) * indexedStructureSize));


  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16*1024*1024);
  /**
 **************************************************************************
 * Start Indexing first
 **************************************************************************
 */
  gpuErrchk(cudaDeviceSynchronize());

  

  INDEXING_STRUCTURE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_dataset, d_indexTreeMetaData, d_minPoints, d_binWidth, d_results,
      d_indexBuckets, d_dataKey, d_dataValue, d_upperBounds);
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

  /**
 **************************************************************************
 * Start the DBSCAN algorithm
 **************************************************************************
 */

  // Keep track of number of cluster formed without global merge
  int runningCluster = THREAD_BLOCKS;
  // Global cluster count
  int clusterCount = 0;

  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to conmtrol the while loop
  bool exit = false;

  clock_t monitorStart, monitorStop, dbscanKernelStart, dbscanKernelStop;
  float monitorTime = 0.0;
  float dbscanKernelTime = 0.0;
  float mergeTime = 0.0;

  while (!exit) {

    monitorStart = clock();
    // Monitor the seed list and return the comptetion status of points
    int completed =
        MonitorSeedPoints(unprocessedPoints, &runningCluster,
                          d_cluster, d_seedList, d_seedLength,
                          d_collisionMatrix, d_extraCollision, d_results, &mergeTime);

    // printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
    //     unprocessedPoints.size());

    monitorStop = clock();
    monitorTime += (float)(monitorStop - monitorStart) / CLOCKS_PER_SEC;

    // If all points are processed, exit
    if (completed) {
      exit = true;
    }

    if (exit) break;

    dbscanKernelStart = clock();

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_extraCollision, d_results, d_indexBuckets, d_indexesStack,
        d_dataValue, d_upperBounds, d_binWidth);
    gpuErrchk(cudaDeviceSynchronize());

    dbscanKernelStop = clock();
    dbscanKernelTime += (float)(dbscanKernelStop - dbscanKernelStart) / CLOCKS_PER_SEC;
  }

  /**
 **************************************************************************
 * End DBSCAN and show the results
 **************************************************************************
 */
  totalTimeStop = clock();
  
  // Get the DBSCAN result
  GetDbscanResult(d_cluster, &runningCluster, &clusterCount, &noiseCount);
  
  totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
  indexingTime = (float)(indexingStop - indexingStart) / CLOCKS_PER_SEC;

  printf("==============================================\n");
  printf("Dataset: %d\nEPS: %f\nMINPTS: %d\n", DATASET_COUNT, EPS, MINPTS);
  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);
  printf("=======\n");
  printf("Indexing Time: %3.2f seconds\n", indexingTime);
  printf("Merge Time: %3.2f seconds\n", mergeTime);
  printf("Communication Time: %3.2f seconds\n", monitorTime - mergeTime);
  printf("DBSCAN kernel Time: %3.2f seconds\n", dbscanKernelTime);
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
  cudaFree(d_upperBounds);
  cudaFree(d_binWidth);
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
    double tmp;
    sscanf(field, "%lf", &tmp);
    dataset[cnt] = tmp;
    cnt++;

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        double tmp;
        sscanf(field, "%lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;
      }
    }
  }
  fclose(fp);
  return 0;
}

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


__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *results,
                       struct IndexStructure **indexBuckets,

                       int *indexesStack, int *dataValue, double *upperBounds,
                       double *binWidth) {
  // Point ID to expand by a block
  __shared__ int pointID;

  // Neighbors to store of neighbors points exceeds minpoints
  __shared__ int neighborBuffer[64];

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
                       int *d_results, float *mergeTime) {
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


  clock_t mergeStart, mergeStop;

  mergeStart = clock();

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

  mergeStop = clock();

  *mergeTime += (float)(mergeStop - mergeStart) / CLOCKS_PER_SEC;

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
  


  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_localCluster, sizeof(int) * DATASET_COUNT,
                        cudaMemcpyDeviceToHost));
  ofstream outputFile;
  outputFile.open("./out/cuda_dclust_extended.txt");
  for (int j = 0; j < DATASET_COUNT; j++) {
    outputFile << localCluster[j] << endl;
  }
  outputFile.close();
  free(localCluster);


  cudaFree(d_localCluster);
}



int main() {

  // Generate random datasets
  char *datasetPath;
  double setOfR[5];
  int setOfMinPts[5];
  int defaultMin, defaultPts;
  double defaultR;
  int setOfDataSize[5];

  if (PORTO) {
    setOfDataSize[0] = 40000;
    setOfDataSize[1] = 80000;
    setOfDataSize[2] = 160000;
    setOfDataSize[3] = 320000;
    setOfDataSize[4] = 640000;

    setOfR[0] = 0.002;
    setOfR[1] = 0.004;
    setOfR[2] = 0.006;
    setOfR[3] = 0.008;
    setOfR[4] = 0.01;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.008;

    defaultPts = 160000;

    datasetPath = "/data/dbscan/Porto_taxi_data.csv";
  }

  if (NGSI) {
    setOfDataSize[0] = 50000;
    setOfDataSize[1] = 100000;
    setOfDataSize[2] = 200000;
    setOfDataSize[3] = 400000;
    setOfDataSize[4] = 800000;

    setOfR[0] = 0.2;
		setOfR[1] = 0.4;
		setOfR[2] = 0.6;
		setOfR[3] = 0.8;
		setOfR[4] = 1;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.8;

    defaultPts = 400000;

    datasetPath = "/data/dbscan/NGSIM_Data.txt";
  }

  if (SPATIAL) {
    setOfDataSize[0] = 25000;
    setOfDataSize[1] = 50000;
    setOfDataSize[2] = 100000;
    setOfDataSize[3] = 200000;
    setOfDataSize[4] = 400000;

    setOfR[0] = 0.002;
    setOfR[1] = 0.004;
    setOfR[2] = 0.006;
    setOfR[3] = 0.008;
    setOfR[4] = 0.01;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.008;

    defaultPts = 400000;

    datasetPath = "/home/mpoudel/datasets/3D_spatial_network.csv";
  }

  if (IONO2D) {
    setOfDataSize[0] = 50000;
    setOfDataSize[1] = 100000;
    setOfDataSize[2] = 200000;
    setOfDataSize[3] = 400000;
    setOfDataSize[4] = 800000;

    setOfR[0] = 0.5;
    setOfR[1] = 0.75;
    setOfR[2] = 1;
    setOfR[3] = 1.25;
    setOfR[4] = 1.5;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 4;
    defaultR = 1.5;

    defaultPts = 400000;

    datasetPath = "/data/geodata/iono_20min_2Mpts_2D.txt";
  }

    // // Different set of Eps
    // printf("################ EPS IMPACT ################\n");
    // for (int i = 0; i < 5; i++) {
    //   runDBSCAN(datasetPath, defaultPts, setOfR[i], defaultMin);
    // }

    // // Different set of MinPts
    // printf("################ MINPTS IMPACT ################\n");
    // for (int i = 0; i < 5; i++) {
    //   runDBSCAN(datasetPath, defaultPts, defaultR, setOfMinPts[i]);
    // }
  
    // Different set of Points
    printf("################ POINTS IMPACT ################\n");
    for (int i = 0; i < 5; i++) {
      runDBSCAN(datasetPath, setOfDataSize[i], defaultR, defaultMin);
    }

  
}