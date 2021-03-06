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
#define THREAD_BLOCKS 1024

// Number of threads per block
#define THREAD_COUNT 512

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

#define POINTS_SEARCHED 800000

#define PARTITION_SIZE 100

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
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_results);

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
  int runningCluster = 0;

  // Global cluster count
  int clusterCount = 0;

  // Keeps track of number of noises
  int noiseCount = 0;

  // Handler to conmtrol the while loop
  bool exit = false;

  while (!exit) {
    // Monitor the seed list and return the comptetion status of points
    int completed = MonitorSeedPoints(
        unprocessedPoints, &runningCluster, d_cluster, d_seedList, d_seedLength,
        d_collisionMatrix, d_extraCollision, d_results);

    // printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
    //        unprocessedPoints.size());

    // If all points are processed, exit
    if (completed) {
      exit = true;
    }

    if (exit) break;

    // Kernel function to expand the seed list
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_extraCollision, d_results, d_indexBuckets, d_indexesStack,
        d_dataValue);
    gpuErrchk(cudaDeviceSynchronize());
  }

  /**
   **************************************************************************
   * End DBSCAN and show the results
   **************************************************************************
   */

  // Get the DBSCAN result
  GetDbscanResult(d_dataset, d_cluster, &runningCluster, &clusterCount,
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

bool MonitorSeedPoints(vector<int> &unprocessedPoints, int *runningCluster,
                       int *d_cluster, int *d_seedList, int *d_seedLength,
                       int *d_collisionMatrix, int *d_extraCollision,
                       int *d_results) {
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
      cudaMemset(d_results, -1, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

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

  int *localExtraCollision;
  localExtraCollision =
      (int *)malloc(sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE);
  gpuErrchk(cudaMemcpy(localExtraCollision, d_extraCollision,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE,
                       cudaMemcpyDeviceToHost));

  /**
   **************************************************************************
   * If seedlist is empty and refill is also empty Then check the `
   * between chains and finalize the clusters
   **************************************************************************
   */

  // Define cluster to map the collisions
  map<int, int> clusterMap;
  set<int> blockSet;

  // Insert chains in blockset
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    blockSet.insert(i);
  }

  set<int>::iterator it;

  // Iterate through the block set until it's empty
  while (blockSet.empty() == 0) {
    // Get a chain from blockset
    it = blockSet.begin();
    int curBlock = *it;

    // Expansion Queue is use to see expansion of collision
    set<int> expansionQueue;

    // Final Queue stores mapped chains for blockset chain
    set<int> finalQueue;

    // Insert current chain from blockset to expansion and final queue
    expansionQueue.insert(curBlock);
    finalQueue.insert(curBlock);

    // Iterate through expansion queue until it's empty
    while (expansionQueue.empty() == 0) {
      // Get first element from expansion queue
      it = expansionQueue.begin();
      int expandBlock = *it;

      // Remove the element because we are about to expand
      expansionQueue.erase(it);

      // Also erase from blockset, because we checked this chain
      blockSet.erase(expandBlock);

      // Loop through chains to see more collisions
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;

        // If there is collision, insert the chain in finalqueue
        // Also, insert in expansion queue for further checking
        // of collision with this chain
        if (localCollisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    }

    // Iterate through final queue, and map collided chains with blockset chain
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

  // Check extra collision with cluster ID greater than thread block
  vector<vector<int>> localClusterMerge(THREAD_BLOCKS, vector<int>());
  for (int i = 0; i < THREAD_BLOCKS; i++) {
    for (int j = 0; j < EXTRA_COLLISION_SIZE; j++) {
      if (localExtraCollision[i * EXTRA_COLLISION_SIZE + j] == UNPROCESSED)
        break;
      bool found = find(localClusterMerge[clusterMap[i]].begin(),
                        localClusterMerge[clusterMap[i]].end(),
                        localExtraCollision[i * EXTRA_COLLISION_SIZE + j]) !=
                   localClusterMerge[clusterMap[i]].end();

      if (!found &&
          localExtraCollision[i * EXTRA_COLLISION_SIZE + j] >= THREAD_BLOCKS) {
        localClusterMerge[clusterMap[i]].push_back(
            localExtraCollision[i * EXTRA_COLLISION_SIZE + j]);
      }
    }
  }

  // Check extra collision with cluster ID greater than thread block
  for (int i = 0; i < localClusterMerge.size(); i++) {
    if (localClusterMerge[i].empty()) continue;
    for (int j = 0; j < localClusterMerge[i].size(); j++) {
      for (int k = 0; k < DATASET_COUNT; k++) {
        if (localCluster[k] == localClusterMerge[i][j]) {
          localCluster[k] = localClusterMerge[clusterMap[i]][0];
        }
      }
    }

    // Also, Assign the mapped chains to the first cluster in extra collision
    for (int x = 0; x < clustersList[clusterMap[i]].size(); x++) {
      localCluster[clustersList[clusterMap[i]][x]] =
          localClusterMerge[clusterMap[i]][0];
    }

    // Clear the mapped chains, as we assigned to clsuter already
    clustersList[clusterMap[i]].clear();
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

  gpuErrchk(cudaMemset(d_extraCollision, -1,
                       sizeof(int) * THREAD_BLOCKS * EXTRA_COLLISION_SIZE));

  /**
   **************************************************************************
   * Free CPU memory allocations
   **************************************************************************
   */

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

  for (int j = 0; j < DATASET_COUNT; j++) {
    if (finalClusterMap[localCluster[j]] >= 0) {
      localCluster[j] = finalClusterMap[localCluster[j]];
    } else {
      localCluster[j] = 0;
    }
  }

  /*
  // Output to file
  ofstream outputFile;
  outputFile.open("./out/gpu_dbscan_output.txt");
  for (int j = 0; j < DATASET_COUNT; j++) {
    outputFile << localCluster[j] << endl;
  }
  outputFile.close();

  */

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

  searchPoints(point, chainID, dataset, results, indexBuckets, indexesStack,
               dataValue);

  __syncthreads();

  for (int i = threadIdx.x; i < POINTS_SEARCHED; i = i + THREAD_COUNT) {
    int nearestPoint = results[chainID * POINTS_SEARCHED + i];

    if (nearestPoint == -1) break;

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
                        collisionMatrix, extraCollision);
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
                      collisionMatrix, extraCollision);
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
  for (int i = 0; i < DIMENSION; i++) {
    indexConstruction(i, indexTreeMetaData, minPoints, binWidth, indexBuckets);
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
  for (int k = blockIdx.x + indexTreeMetaData[level * RANGE + 0];
       k < indexTreeMetaData[level * RANGE + 1]; k = k + THREAD_BLOCKS) {
         if(threadIdx.x == 0) {
           indexBuckets[k]->dimension = level;
         }
         __syncthreads();
    for (int i = threadIdx.x; i < PARTITION_SIZE; i = i + THREAD_COUNT) {
      int currentBucketIndex =
          indexTreeMetaData[level * RANGE + 1] + i +
          (k - indexTreeMetaData[level * RANGE + 0]) * PARTITION_SIZE;

      
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
  __shared__ int matchedIndex[50];
  __shared__ int matchedIndexCount;
  __shared__ int resultsCount;
  __shared__ int indexBucketSize;
  __shared__ int currentIndex;
  __shared__ int currentIndexSize;
  __shared__ int stopTraverse;
  __shared__ double comparingData;

  if (threadIdx.x == 0) {
    matchedIndexCount = 0;
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
    if(indexBuckets[currentIndex]->dimension >= DIMENSION) break;
    if (threadIdx.x == 0) {
      currentIndexSize = currentIndexSize - 1;
      currentIndex = indexesStack[currentIndexSize];
      stopTraverse = 0;
      comparingData = data[indexBuckets[currentIndex]->dimension];
    }
    __syncthreads();

    for (int k = threadIdx.x; k < PARTITION_SIZE; k = k + THREAD_COUNT) {
      if (stopTraverse == 1) break;

      
      double leftRange =
          indexBuckets[indexBuckets[currentIndex]->childBuckets[k]]->range[0];
      double rightRange =
          indexBuckets[indexBuckets[currentIndex]->childBuckets[k]]->range[1];

      if (comparingData >= leftRange && comparingData < rightRange) {
        if (indexBuckets[currentIndex]->dimension == DIMENSION - 1) {
          int oldMatchedIndexCount = atomicAdd(&matchedIndexCount, 1);
          matchedIndex[oldMatchedIndexCount] =
              indexBuckets[currentIndex]->childBuckets[k];
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
  printf("%d\n", matchedIndexCount);
  for (int x = 0; x < matchedIndexCount; x++) {
    for (int i = threadIdx.x + indexBuckets[matchedIndex[x]]->dataBegin;
         i < indexBuckets[matchedIndex[x]]->dataEnd; i = i + THREAD_COUNT) {
      int resultCountCurrent = atomicAdd(&resultsCount, 1);
      if(resultCountCurrent < POINTS_SEARCHED) {
        results[chainID * POINTS_SEARCHED + resultCountCurrent] = dataValue[i];
      }
    }

    if (matchedIndex[x] > 0) {
      for (int i = threadIdx.x + indexBuckets[matchedIndex[x] - 1]->dataBegin;
           i < indexBuckets[matchedIndex[x] - 1]->dataEnd;
           i = i + THREAD_COUNT) {
        int resultCountCurrent = atomicAdd(&resultsCount, 1);
        if(resultCountCurrent < POINTS_SEARCHED) {
          results[chainID * POINTS_SEARCHED + resultCountCurrent] = dataValue[i];
        }
      }
    }

    if (matchedIndex[x] < PARTITION_SIZE - 1) {
      for (int i = threadIdx.x + indexBuckets[matchedIndex[x] + 1]->dataBegin;
           i < indexBuckets[matchedIndex[x] + 1]->dataEnd;
           i = i + THREAD_COUNT) {
        int resultCountCurrent = atomicAdd(&resultsCount, 1);
        if(resultCountCurrent < POINTS_SEARCHED) {
          results[chainID * POINTS_SEARCHED + resultCountCurrent] = dataValue[i];
        }
        
      }
    }
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