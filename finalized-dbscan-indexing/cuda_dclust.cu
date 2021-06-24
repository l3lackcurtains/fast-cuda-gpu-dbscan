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
#include <set>
#include <vector>

using namespace std;

#define RANGE 2
#define UNPROCESSED -1
#define NOISE -2

#define DIMENSION 2
#define TREE_LEVELS (DIMENSION + 1)

#define THREAD_BLOCKS 512
#define THREAD_COUNT 512

#define MAX_SEEDS 2048
#define EXTRA_COLLISION_SIZE 512

// #define DATASET_COUNT 1864620
#define DATASET_COUNT 200000

#define MINPTS 4
#define EPS 1.5

#define PARTITION_SIZE 80
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
* Declare CPU and GPU Functions
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int ImportDataset(char const *fname, double *dataset);

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix);

void GetDbscanResult(double *d_dataset, int *d_cluster, int *runningCluster,
                     int *clusterCount, int *noiseCount);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *neighborsPoints, int *maxSize);

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,

                                int *collisionMatrix);
/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Indexing data structure and functions
//////////////////////////////////////////////////////////////////////////
**/

// struct IndexStructure {
//   int level;
//   double range[2];
//   struct IndexStructure *buckets[PARTITION];
//   int datas[POINTS_SEARCHED];
// };

struct IndexStructure {
  int dimension;
  int dataBegin;
  int dataEnd;
  int childFrom;
};

void indexConstruction(double *dataset, int *indexTreeMetaData,
                       double *minPoints, double *binWidth,
                       struct IndexStructure **indexBuckets, int *dataKey,
                       int *dataValue, double *upperBounds);

void insertData(int id, double *dataset, struct IndexStructure **indexBuckets,
                int *dataKey, int *dataValue, double *upperBounds,
                double *binWidth);

void searchPoints(double *data, int chainID, double *dataset, int *results,
                  struct IndexStructure **indexBuckets, int *dataValue,
                  double *upperBounds, double *binWidth);

/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Main CPU function
//////////////////////////////////////////////////////////////////////////
**************************************************************************
*/
int main(int argc, char **argv) {
  /**
   **************************************************************************
   * Get the dataset file from argument and import data
   **************************************************************************
   */

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
  for (int i = 0; i < 4; i++) {
    printf("Sample Data %f\n", importedDataset[i]);
  }

  // Get the total count of dataset
  vector<int> unprocessedPoints;
  for (int x = DATASET_COUNT - 1; x >= 0; x--) {
    unprocessedPoints.push_back(x);
  }

  printf("Preprocessed %lu data in dataset\n", unprocessedPoints.size());

  // Reset the GPU device for potential memory issues
  gpuErrchk(cudaDeviceReset());
  gpuErrchk(cudaFree(0));

  // Start the time
  clock_t totalTimeStart, totalTimeStop, indexingStart, indexingStop;
  double totalTime = 0.0;
  double indexingTime = 0.0;
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

  gpuErrchk(cudaMalloc((void **)&d_dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_cluster, sizeof(int) * DATASET_COUNT));

  gpuErrchk(cudaMalloc((void **)&d_seedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS));

  gpuErrchk(cudaMalloc((void **)&d_seedLength, sizeof(int) * THREAD_BLOCKS));

  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));


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


  /**
   **************************************************************************
   * Index construction
   **************************************************************************
   */

  indexingStart = clock();

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

  printf("==============================================\n");

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

  for (int i = 0; i < TREE_LEVELS; i++) {
    printf("#%d ", i);
    printf("Partition: %d ", treeLevelPartition[i]);
    printf("Range: %d %d\n", startEndIndexes[i * RANGE + 0],
           startEndIndexes[i * RANGE + 1]);
  }
  printf("==============================================\n");

  int indexedStructureSize = startEndIndexes[DIMENSION * RANGE + 1];

  printf("Index Structure Size: %lf GB.\n",
         (sizeof(struct IndexStructure) * indexedStructureSize) /
             (1024 * 1024 * 1024.0));

  struct IndexStructure **indexBuckets;
  indexBuckets = (struct IndexStructure **)malloc(
      sizeof(struct IndexStructure *) * indexedStructureSize);

  for (int x = 0; x < indexedStructureSize; x++) {
    indexBuckets[x] =
        (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
  }

  int *dataKey;
  int *dataValue;
  double *upperBounds;

  dataKey = (int *)malloc(sizeof(int) * DATASET_COUNT);
  dataValue = (int *)malloc(sizeof(int) * DATASET_COUNT);
  upperBounds = (double *)malloc(sizeof(double) * indexedStructureSize);

  indexConstruction(importedDataset, startEndIndexes, minPoints, binWidth,
                    indexBuckets, dataKey, dataValue, upperBounds);

  indexingStop = clock();

  /**
   **************************************************************************
   * Start the DBSCAN algorithm
   **************************************************************************
   */

  // Keep track of number of cluster formed without global merge
  int runningCluster = 0;

  // Global cluster count
  int clusterCount = 0;

  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to conmtrol the while loop
  bool exit = false;

  int *d_maxSize;
  gpuErrchk(cudaMalloc((void **)&d_maxSize, sizeof(int)));

  int *results = (int *)malloc(sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED);

  while (!exit) {
    // Monitor the seed list and return the comptetion status of points
    int completed = MonitorSeedPoints(unprocessedPoints, &runningCluster,
                                      d_cluster, d_seedList, d_seedLength,
                                      d_collisionMatrix);
    // printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
    //        unprocessedPoints.size());

    // If all points are processed, exit
    if (completed) {
      exit = true;
    }

    if (exit) break;

    int *localSeedLength;
    localSeedLength = (int *)malloc(sizeof(int) * THREAD_BLOCKS);
    gpuErrchk(cudaMemcpy(localSeedLength, d_seedLength,
                         sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost));

    int *localSeedList;
    localSeedList = (int *)malloc(sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
    gpuErrchk(cudaMemcpy(localSeedList, d_seedList,
                         sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < THREAD_BLOCKS; i++) {
      for (int k = 0; k < POINTS_SEARCHED; k++) {
        results[i * POINTS_SEARCHED + k] = -1;
      }

      if (localSeedLength[i] == 0) continue;

      int seedPointId = localSeedList[i * MAX_SEEDS + localSeedLength[i] - 1];
      double *searchPoint = (double *)malloc(sizeof(double) * DIMENSION);

      for (int j = 0; j < DIMENSION; j++) {
        searchPoint[j] = importedDataset[seedPointId * DIMENSION + j];
      }

      searchPoints(searchPoint, i, importedDataset, results, indexBuckets,
                   dataValue, upperBounds, binWidth);
    }

    int maxSize = 0;
    for (int i = 0; i < THREAD_BLOCKS; i++) {
      int count = 0;
      for (int j = 0; j < POINTS_SEARCHED; j++) {
        if (results[i * POINTS_SEARCHED + j] == -1) continue;

        count += indexBuckets[results[i * POINTS_SEARCHED + j]]->dataEnd -
                 indexBuckets[results[i * POINTS_SEARCHED + j]]->dataBegin;
      }

      if (count > maxSize) {
        maxSize = count;
      }
    }

    gpuErrchk(
        cudaMemcpy(d_maxSize, &maxSize, sizeof(int), cudaMemcpyHostToDevice));

    int *neighborsPoints = (int *)malloc(sizeof(int) * THREAD_BLOCKS * maxSize);

    for (int x = 0; x < THREAD_BLOCKS * maxSize; x++) {
      neighborsPoints[x] = -1;
    }

    for (int i = 0; i < THREAD_BLOCKS; i++) {
      int dataCount = 0;
      for (int j = 0; j < POINTS_SEARCHED; j++) {
        if (results[i * POINTS_SEARCHED + j] == -1) continue;
        for (int x = indexBuckets[results[i * POINTS_SEARCHED + j]]->dataBegin;
             x < indexBuckets[results[i * POINTS_SEARCHED + j]]->dataEnd; x++) {
          neighborsPoints[i * maxSize + dataCount++] = dataValue[x];
        }
      }
    }

    int *d_neighborsPoints;
    gpuErrchk(cudaMalloc((void **)&d_neighborsPoints,
                         sizeof(int) * THREAD_BLOCKS * maxSize));

    gpuErrchk(cudaMemcpy(d_neighborsPoints, neighborsPoints,
                         sizeof(int) * THREAD_BLOCKS * maxSize,
                         cudaMemcpyHostToDevice));

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_neighborsPoints, d_maxSize);
    gpuErrchk(cudaDeviceSynchronize());

    free(localSeedList);
    free(localSeedLength);
    free(neighborsPoints);
    cudaFree(d_neighborsPoints);
  }

  /**
   **************************************************************************
   * End DBSCAN and show the results
   **************************************************************************
   */

  // Get the DBSCAN result
  GetDbscanResult(d_dataset, d_cluster, &runningCluster, &clusterCount,
                  &noiseCount);

  printf("==============================================\n");
  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);
  printf("==============================================\n");

  totalTimeStop = clock();
  totalTime = (double)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
  indexingTime = (double)(indexingStop - indexingStart) / CLOCKS_PER_SEC;
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
  cudaFree(d_maxSize);

  free(results);
  free(dataKey);
  free(dataValue);
  for (int x = 0; x < indexedStructureSize; x++) {
    free(indexBuckets[x]);
  }
  free(indexBuckets);
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

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix) {
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

  /**
   **************************************************************************
   * Check if the seedlist is not empty, If so continue with DBSCAN process
   * if seedlist is empty, check refill seed list
   * if there are points in refill list, transfer to seedlist
   **************************************************************************
   */

  int completeSeedListFirst = false;

  // Check if the seed list is empty
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    // If seed list is not empty set completeSeedListFirst as true
    if (localSeedLength[i] > 0) {
      completeSeedListFirst = true;
    }
  }

  /**
   **************************************************************************
   * If seedlist still have points, go to DBSCAN process
   **************************************************************************
   */

  if (completeSeedListFirst) {
    free(localSeedList);
    free(localSeedLength);
    return false;
  }

  /**
   **************************************************************************
   * Copy GPU variables to CPU variables for collision detection
   **************************************************************************
   */

  int *localCluster;
  localCluster = (int *)malloc(sizeof(int) * DATASET_COUNT);
  gpuErrchk(cudaMemcpy(localCluster, d_cluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyDeviceToHost));

  int *localCollisionMatrix;
  localCollisionMatrix =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS);
  gpuErrchk(cudaMemcpy(localCollisionMatrix, d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyDeviceToHost));

  /**
   **************************************************************************
   * If seedlist is empty and refill is also empty Then check the `
   * between chains and finalize the clusters
   **************************************************************************
   */
  map<int, int> clusterMap;
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

  // Loop through dataset and get points for mapped chain
  vector<vector<int>> clustersList(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < DATASET_COUNT; i++) {
    if (localCluster[i] >= 0 && localCluster[i] < THREAD_BLOCKS) {
      clustersList[clusterMap[localCluster[i]]].push_back(i);
    }
  }

  // From all the mapped chains, form a new cluster
  for (int i = 0; i < clustersList.size(); i++) {
    if (clustersList[i].size() == 0) continue;
    for (int x = 0; x < clustersList[i].size(); x++) {
      localCluster[clustersList[i][x]] = *runningCluster + THREAD_BLOCKS;
    }
    (*runningCluster)++;
  }

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

  gpuErrchk(cudaMemcpy(d_cluster, localCluster, sizeof(int) * DATASET_COUNT,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_seedLength, localSeedLength,
                       sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_seedList, localSeedList,
                       sizeof(int) * THREAD_BLOCKS * MAX_SEEDS,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(d_collisionMatrix, -1,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));


  /**
   **************************************************************************
   * Free CPU memory allocations
   **************************************************************************
   */

  free(localCluster);
  free(localSeedList);
  free(localSeedLength);
  free(localCollisionMatrix);

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
  for (int i = THREAD_BLOCKS; i <= (*runningCluster) + THREAD_BLOCKS; i++) {
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

  free(localCluster);
  free(dataset);
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
                       int *neighborsPoints,
                       int *maxSize) {
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

  // Length of the seedlist to check its size
  __shared__ int currentSeedLength;

  /**
   **************************************************************************
   * Get current chain length, and If its zero, exit
   **************************************************************************
   */

  // Assign chainID, current seed length and pointID
  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
    currentSeedLength = seedLength[chainID];
    pointID = seedList[chainID * MAX_SEEDS + currentSeedLength - 1];
  }
  __syncthreads();

  // If seed length is 0, return
  if (currentSeedLength == 0) return;

  // Check if the point is already processed
  if (threadIdx.x == 0) {
    seedLength[chainID] = currentSeedLength - 1;
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

  for (int i = threadIdx.x; i < maxSize[0]; i = i + THREAD_COUNT) {
    int nearestPoint = neighborsPoints[chainID * maxSize[0] + i];
    if (nearestPoint == -1) continue;

    register double comparingPoint[DIMENSION];
    for (int x = 0; x < DIMENSION; x++) {
      comparingPoint[x] = dataset[nearestPoint * DIMENSION + x];
    }

    // find the distance between the points
    register double distance = 0;
    for (int x = 0; x < DIMENSION; x++) {
      distance +=
          (point[x] - comparingPoint[x]) * (point[x] - comparingPoint[x]);
    }

    // If distance is less than elipson, mark point as candidate
    if (distance <= EPS * EPS) {
      register int currentNeighborCount = atomicAdd(&neighborCount, 1);
      if (currentNeighborCount >= MINPTS) {
        MarkAsCandidate(nearestPoint, chainID, cluster, seedList, seedLength,
                        collisionMatrix);
      } else {
        neighborBuffer[currentNeighborCount] = nearestPoint;
      }
    }
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
      MarkAsCandidate(neighborBuffer[i], chainID, cluster, seedList, seedLength,
                      collisionMatrix);
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
                                int *collisionMatrix) {
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

void indexConstruction(double *dataset, int *indexTreeMetaData,
                       double *minPoints, double *binWidth,
                       struct IndexStructure **indexBuckets, int *dataKey,
                       int *dataValue, double *upperBounds) {
  for (int level = 0; level < DIMENSION; level++) {
    for (int k = indexTreeMetaData[level * RANGE + 0];
         k < indexTreeMetaData[level * RANGE + 1]; k++) {
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
  }

  for (int i = 0; i < DATASET_COUNT; i++) {
    insertData(i, dataset, indexBuckets, dataKey, dataValue, upperBounds,
               binWidth);
  }

  thrust::sort_by_key(thrust::host, dataKey, dataKey + DATASET_COUNT,
                      dataValue);

  int indexingRange;
  indexingRange = indexTreeMetaData[DIMENSION * RANGE + 1] -
                  indexTreeMetaData[DIMENSION * RANGE];

  for (int i = 0; i < indexingRange; i++) {
    int idx = indexTreeMetaData[DIMENSION * RANGE] + i;

    thrust::pair<int *, int *> dataPositioned;

    dataPositioned = thrust::equal_range(thrust::host, dataKey,
                                         dataKey + DATASET_COUNT, idx);

    indexBuckets[idx]->dataBegin = dataPositioned.first - dataKey;
    indexBuckets[idx]->dataEnd = dataPositioned.second - dataKey;
  }
}

void insertData(int id, double *dataset, struct IndexStructure **indexBuckets,
                int *dataKey, int *dataValue, double *upperBounds,
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

    int k =
        thrust::upper_bound(
            thrust::host, upperBounds + indexBuckets[currentIndex]->childFrom,
            upperBounds + indexBuckets[currentIndex]->childFrom +
                PARTITION_SIZE,
            comparingData, thrust::less<double>()) -
        upperBounds;

    if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {
      dataValue[id] = id;
      dataKey[id] = k;
      found = true;
    }
    currentIndex = k;
  }
}

void searchPoints(double *data, int chainID, double *dataset, int *results,
                  struct IndexStructure **indexBuckets, int *dataValue,
                  double *upperBounds, double *binWidth) {
  int resultsCount;
  int indexBucketSize;
  int currentIndex;
  int currentIndexSize;
  double comparingData;

  resultsCount = 0;
  indexBucketSize = 1;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }
  int stackSize = indexBucketSize * THREAD_BLOCKS;
  int *indexesStack = (int *)malloc(sizeof(int) * stackSize);

  indexBucketSize = indexBucketSize * chainID;
  currentIndexSize = indexBucketSize;

  indexesStack[currentIndexSize++] = 0;

  while (currentIndexSize > indexBucketSize) {
    currentIndexSize = currentIndexSize - 1;
    currentIndex = indexesStack[currentIndexSize];
    comparingData = data[indexBuckets[currentIndex]->dimension];

    for (int k = indexBuckets[currentIndex]->childFrom;
         k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE; k++) {
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
          results[chainID * POINTS_SEARCHED + resultsCount++] = k;

          if (k > indexBuckets[currentIndex]->childFrom) {
            results[chainID * POINTS_SEARCHED + resultsCount++] = k - 1;
          }

          if (k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE - 1) {
            results[chainID * POINTS_SEARCHED + resultsCount++] = k + 1;
          }

        } else {
          indexesStack[currentIndexSize++] = k;

          if (k > indexBuckets[currentIndex]->childFrom) {
            indexesStack[currentIndexSize++] = k - 1;
          }
          if (k < indexBuckets[currentIndex]->childFrom + PARTITION_SIZE - 1) {
            indexesStack[currentIndexSize++] = k + 1;
          }
        }
      }
    }
  }

  free(indexesStack);
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