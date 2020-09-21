#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <time.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <bits/stdc++.h>

using namespace std;

#define PARTITION 240

// Number of data in dataset to use
#define DATASET_COUNT 100000
// #define DATASET_COUNT 1864620

// Dimension of the dataset
#define DIMENSION 2

// Maximum size of seed list
#define MAX_SEEDS 1024

// Extra collission size to detect final clusters collision
#define EXTRA_COLLISION_SIZE 128

// Number of blocks
#define THREAD_BLOCKS 64

// Number of threads per block
#define THREAD_COUNT 128

// Status of points that are not clusterized
#define UNPROCESSED -1

// Status for noise point
#define NOISE -2

// Minimum number of points in DBSCAN
#define MINPTS 4

// Epslion value in DBSCAN
#define EPS 1.5

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
                       int *d_collisionMatrix, int *d_extraCollision);

void GetDbscanResult(double *d_dataset, int *d_cluster, int *runningCluster,
                     int *clusterCount, int *noiseCount);

__global__ void DBSCAN(double *dataset, int *cluster, int *seedList,
                       int *seedLength, int *collisionMatrix,
                       int *extraCollision, int *neighborsPoints, int * maxSize);

__device__ void MarkAsCandidate(int neighborID, int chainID, int *cluster,
                                int *seedList, int *seedLength,

                                int *collisionMatrix, int *extraCollision);
/**
**************************************************************************
//////////////////////////////////////////////////////////////////////////
* Indexing data structure and functions
//////////////////////////////////////////////////////////////////////////
**/

struct dataNode {
    int id;
    struct dataNode *child;
};

struct IndexStructure {
    int level;
    double range[2];
    struct IndexStructure *buckets[PARTITION];
    struct dataNode *dataRoot;
};

void indexConstruction(double *dataset, struct IndexStructure *indexRoot, int *partition, double minPoints[DIMENSION]);

void insertData(int id, double *data, struct IndexStructure *indexRoot, int *partition);

vector<int> searchPoints(double *data, struct IndexStructure *indexRoot, int *partition);

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
   * Index construction
   **************************************************************************
   */
  
   struct IndexStructure *indexRoot = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

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

    int *partition = (int *)malloc(sizeof(int) * DIMENSION);

    for (int i = 0; i < DIMENSION; i++) {
        partition[i] = 0;
        double curr = minPoints[i];
        while (curr < maxPoints[i]) {
            partition[i]++;
            curr += EPS;
        }
    }

    cout<<partition[0]<<" "<<partition[1]<<endl;

    indexConstruction(importedDataset, indexRoot, partition, minPoints);

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

  int *d_neighborsPoints;

  int *d_maxSize;
  gpuErrchk(cudaMalloc((void **)&d_maxSize, sizeof(int)));

  while (!exit) {
    // Monitor the seed list and return the comptetion status of points
    int completed = MonitorSeedPoints(unprocessedPoints, &runningCluster,
                                      d_cluster, d_seedList, d_seedLength,
                                      d_collisionMatrix, d_extraCollision);
    printf("Running cluster %d, unprocessed points: %lu\n", runningCluster,
           unprocessedPoints.size());
    
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

    vector<vector<int>> pointsList(THREAD_BLOCKS, vector<int>());
    for (int i = 0; i < THREAD_BLOCKS; i++) {
      if (localSeedLength[i] == 0) continue;

      int seedPointId = localSeedList[i * MAX_SEEDS + localSeedLength[i] - 1];
      double searchPoint[DIMENSION];

      for (int j = 0; j < DIMENSION; j++) {
        searchPoint[j] = importedDataset[seedPointId * DIMENSION + j];
      }
      pointsList[i] = searchPoints(searchPoint, indexRoot, partition);
    }

    int maxSize = 0;
    for (int i = 0; i < pointsList.size(); i++) {
      if (pointsList[i].size() > maxSize) {
        maxSize = pointsList[i].size();
      }
    }
    
    gpuErrchk(
        cudaMemcpy(d_maxSize, &maxSize, sizeof(int), cudaMemcpyHostToDevice));

    int *h_neighborsPoints =
        (int *)malloc(sizeof(int) * THREAD_BLOCKS * maxSize);

    for (int i = 0; i < THREAD_BLOCKS; i++) {
      for (int j = 0; j < maxSize; j++) {
        if(j < pointsList[i].size()) {
           h_neighborsPoints[i * maxSize + j] = pointsList[i][j];
        } else {
           h_neighborsPoints[i * maxSize + j] = -1;
        }
       
      }
    }

    gpuErrchk(cudaMalloc((void **)&d_neighborsPoints,
                         sizeof(int) * THREAD_BLOCKS * maxSize));

    gpuErrchk(cudaMemcpy(d_neighborsPoints, h_neighborsPoints,
                         sizeof(int) * THREAD_BLOCKS * maxSize,
                         cudaMemcpyHostToDevice));

    free(localSeedList);
    free(localSeedLength);
    free(h_neighborsPoints);

    // Kernel function to expand the seed list
    gpuErrchk(cudaDeviceSynchronize());
    DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
        d_dataset, d_cluster, d_seedList, d_seedLength, d_collisionMatrix,
        d_extraCollision, d_neighborsPoints, d_maxSize);
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

  printf("==============================================\n");
  printf("Final cluster after merging: %d\n", clusterCount);
  printf("Number of noises: %d\n", noiseCount);
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
  cudaFree(d_neighborsPoints);
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
                       int *d_collisionMatrix, int *d_extraCollision) {
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

  // for (int i = 0; i < THREAD_BLOCKS; i++) {
  //   printf("%d: ", i);
  //   for (int j = 0; j < localClusterMerge[i].size(); j++) {
  //     printf("%d, ", localClusterMerge[i][j]);
  //   }
  //   printf("\n");
  // }

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

  // Output to file
  ofstream outputFile;
  outputFile.open("./out/gpu_dbscan_output.txt");

  for (int j = 0; j < DATASET_COUNT; j++) {
    if (finalClusterMap[localCluster[j]] >= 0) {
      localCluster[j] = finalClusterMap[localCluster[j]];
    } else {
      localCluster[j] = 0;
    }
  }

  for (int j = 0; j < DATASET_COUNT; j++) {
    outputFile << localCluster[j] << endl;
  }

  outputFile.close();

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
                       int *extraCollision, int *neighborsPoints, int * maxSize) {
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

  for (int i = threadIdx.x; i < (*maxSize); i = i + THREAD_COUNT) {
    int nearestPoint = neighborsPoints[chainID * (*maxSize) + i];
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

void indexConstruction(double *dataset, struct IndexStructure *indexRoot, int *partition, double minPoints[DIMENSION]) {
    stack<struct IndexStructure *> indexStacked;
    indexStacked.push(indexRoot);

    for (int j = 0; j < DIMENSION; j++) {
        stack<struct IndexStructure *> childStacked;

        while (indexStacked.size() > 0) {
            struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentIndex = indexStacked.top();
            indexStacked.pop();
            currentIndex->level = j;

            double rightPoint = minPoints[j] + partition[j] * EPS;

            for (int k = partition[j] - 1; k >= 0; k--) {
                struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

                currentBucket->range[1] = rightPoint;
                rightPoint = rightPoint - EPS;
                currentBucket->range[0] = rightPoint;
                currentBucket->dataRoot = (struct dataNode *)malloc(sizeof(struct dataNode));
                currentBucket->dataRoot->id = -1;

                currentIndex->buckets[k] = currentBucket;
                if (j < DIMENSION - 1) {
                    childStacked.push(currentIndex->buckets[k]);
                }
            }
        }

        while (childStacked.size() > 0) {
            indexStacked.push(childStacked.top());
            childStacked.pop();
        }
    }

    for (int i = 0; i < DATASET_COUNT; i++) {
        double *data =
            (double *)malloc(sizeof(double) * DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            data[j] = dataset[i * DIMENSION + j];
        }
        insertData(i, data, indexRoot, partition);
    }
}

void insertData(int id, double *data, struct IndexStructure *indexRoot, int *partition) {
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

    currentIndex = indexRoot;
    bool found = false;

    while (!found) {
        int dimension = currentIndex->level;
        for (int k = 0; k < partition[dimension]; k++) {
            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentBucket = currentIndex->buckets[k];

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentBucket->range[0];
            float rightRange = (float)currentBucket->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {
                    selectedDataNode = currentBucket->dataRoot;
                    found = true;
                    break;
                }
                currentIndex = currentBucket;
                break;
            }
        }
    }

    if (selectedDataNode->id == -1) {
        selectedDataNode->id = id;
        selectedDataNode->child = (struct dataNode *)malloc(sizeof(struct dataNode));
        selectedDataNode->child->id = -1;
    } else {
        selectedDataNode = selectedDataNode->child;
        while (selectedDataNode->id != -1) {
            selectedDataNode = selectedDataNode->child;
        }
        selectedDataNode->id = id;
        selectedDataNode->child = (struct dataNode *)malloc(sizeof(struct dataNode));
        selectedDataNode->child->id = -1;
    }
}

vector<int> searchPoints(double *data, struct IndexStructure *indexRoot, int *partition) {
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

    vector<struct dataNode *> selectedDataNodes = {};

    vector<struct IndexStructure *> currentIndexes = {};
    currentIndexes.push_back(indexRoot);

    while (!currentIndexes.empty()) {

        currentIndex = currentIndexes.back();
        currentIndexes.pop_back();

        int dimension = currentIndex->level;
        for (int k = 0; k < partition[dimension]; k++) {
            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentBucket = currentIndex->buckets[k];

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentBucket->range[0];
            float rightRange = (float)currentBucket->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {
                    selectedDataNodes.push_back(currentBucket->dataRoot);
                    if (k > 0) {
                        selectedDataNodes.push_back(currentIndex->buckets[k - 1]->dataRoot);
                    }
                    if (k < partition[dimension] - 1) {
                        selectedDataNodes.push_back(currentIndex->buckets[k + 1]->dataRoot);
                    }
                    break;
                }
                currentIndexes.push_back(currentBucket);
                if (k > 0) {
                    currentIndexes.push_back(currentIndex->buckets[k - 1]);
                }
                if (k < partition[dimension] - 1) {
                    currentIndexes.push_back(currentIndex->buckets[k + 1]);
                }
                break;
            }
        }
    }

    vector<int> points = {};
    for (int x = 0; x < selectedDataNodes.size(); x++) {
        selectedDataNode = selectedDataNodes[x];
        while (selectedDataNode->id != -1) {
            points.push_back(selectedDataNode->id);
            selectedDataNode = selectedDataNode->child;
        }
    }

    return points;
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
