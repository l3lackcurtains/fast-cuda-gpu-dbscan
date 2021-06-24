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

 #define DATASET_COUNT 1000

 #define DIMENSION 2
 
 #define MAX_SEEDS 1024
 
 #define THREAD_BLOCKS 128
 #define THREAD_COUNT 128
 
 #define UNPROCESSED -1
 #define NOISE -2

 
 int ImportDataset(char const *fname, float dataset[DATASET_COUNT][DIMENSION]);
 #define cuErrorCheck(ans) { errorAssert((ans), __FILE__, __LINE__); }
 inline void errorAssert(cudaError_t retCode, const char *file, int line)
 {
    if (retCode != cudaSuccess) 
    {
       fprintf(stderr,"GPU op Failed: %s %s %d\n", cudaGetErrorString(retCode), file, line);
       exit(retCode);
    }
 }
 
 /**********************************************
  *  Global GPU data structures and parameters *
  **********************************************/

 __device__ __constant__ int minpts = 4; 
 __device__ __constant__ float eps = 1.5;
 
 __device__ float points[DATASET_COUNT][2];
 
 __device__ int pointState[DATASET_COUNT];
 
 __device__ int seedList[THREAD_BLOCKS][MAX_SEEDS];
 __device__ int curSeedLength[THREAD_BLOCKS];
 
 __device__ int16_t collisionMatrix[THREAD_BLOCKS][THREAD_BLOCKS];
 
 int16_t ** sym_collision;
 int ** sym_seedList;
 int * sym_curSeedLength;
 float ** sym_points;
 int * sym_pointState;
 
 bool SelectNextPointSet(std::vector<int> & pointsRemaining, int * clusterCount);
 void FinalizeClusters(int * states, int * clusterCount);
 __global__ void DBSCAN(void);
 __device__ void markAsCandidate(int pid, int chainID);
 
 int main(int argc, char **argv) {
  char inputFname[500];
  if (argc != 2) {
    fprintf(stderr, "Please provide the dataset file path in the arguments\n");
    exit(0);
  }

  // Get the dataset file name from argument
  strcpy(inputFname, argv[1]);
  printf("Using dataset file %s\n", inputFname);

  float point_gen[DATASET_COUNT][DIMENSION];
    

  // Import data from dataset
  int ret = ImportDataset(inputFname, point_gen);
  if (ret == 1) {
    printf("\nError importing the dataset");
    return 0;
  }

  // Check if the data parsed is correct
  for (int i = 0; i < 2; i++) {
    printf("Sample Data: ");
    for(int x = 0; x < DIMENSION; x++) {
      printf("%f ", point_gen[i][x]);
    }
    printf("\n");
   
  }

  // Get the total count of dataset
  vector<int> pointsRemaining;
  for (int x = 0; x < DATASET_COUNT; x++) {
    pointsRemaining.push_back(x);
  }
     fprintf(stderr,"Generated %lu points\n", pointsRemaining.size());
 
     cuErrorCheck( cudaFree(0) );
 
     cuErrorCheck(cudaGetSymbolAddress((void **)&sym_collision, collisionMatrix));
     cuErrorCheck(cudaGetSymbolAddress((void **)&sym_seedList, seedList));
     cuErrorCheck(cudaGetSymbolAddress((void **)&sym_curSeedLength, curSeedLength));
     cuErrorCheck(cudaGetSymbolAddress((void **)&sym_points, points));
     cuErrorCheck(cudaGetSymbolAddress((void **)&sym_pointState, pointState));
 
     // Initialize CUDA data structures.
     cudaMemset(sym_collision, -1, sizeof(int16_t) * THREAD_BLOCKS * THREAD_BLOCKS);
 
     cudaMemset(sym_seedList, -1, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS);
 
     cudaMemset(sym_curSeedLength, 0, sizeof(int) * THREAD_BLOCKS);
 
     cudaMemset(sym_pointState, UNPROCESSED, sizeof(int) * DATASET_COUNT);
 
     cudaMemcpy(sym_points, point_gen, sizeof(float) * DATASET_COUNT * 2, cudaMemcpyHostToDevice);
 
     int clusterCount = 0;
 
     fprintf(stderr,"Starting DBSCAN\n");
 
     while (SelectNextPointSet(pointsRemaining, &clusterCount) == true) {
         cuErrorCheck(cudaDeviceSynchronize());
 
         DBSCAN<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT,1)>>>();
 
         cuErrorCheck(cudaDeviceSynchronize());
 
         fprintf(stderr, "Points Remaining: %llu\n", pointsRemaining.size());
     }   
 }
 
 bool SelectNextPointSet(std::vector<int> & pointsRemaining, int * clusterCount) {
     int complete = 0;
     bool refresh = true;
 
     int lseedCount[THREAD_BLOCKS];
     memset(lseedCount, 0, sizeof(int) * THREAD_BLOCKS);
 
     cudaMemcpy(lseedCount, sym_curSeedLength, sizeof(int) * THREAD_BLOCKS, cudaMemcpyDeviceToHost);
 
     for (int i = 0; i < THREAD_BLOCKS; i++) {
         if (lseedCount[i] > 0) {
             refresh = false;
             break;
         } 
     }
 
     if (refresh == false)
         return true;
 
     int lpointStates[DATASET_COUNT];
 
     cudaMemcpy(lpointStates, sym_pointState, sizeof(int) * DATASET_COUNT, cudaMemcpyDeviceToHost);
 
     FinalizeClusters(lpointStates, clusterCount);
     
     int pointSeeds[THREAD_BLOCKS][MAX_SEEDS];
 
     cudaMemcpy(pointSeeds, sym_seedList, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS, cudaMemcpyDeviceToHost);
 
     for (int i = 0; i < THREAD_BLOCKS; i++) {
         bool found = false;
         while(!pointsRemaining.empty()) {
             int pos = pointsRemaining.back();
             pointsRemaining.pop_back();
             if ( lpointStates[pos] == UNPROCESSED) {
                 lseedCount[i] = 1;
                 pointSeeds[i][0] = pos;
                 found = true;
                 break;
             }
         }
         if (found == false) {
             complete++;
         }
     }
 
     fprintf(stderr, "Found %d Clusters, %llu points remaining\n", *clusterCount, pointsRemaining.size());
 
     cudaMemcpy(sym_curSeedLength, lseedCount, sizeof(int) * THREAD_BLOCKS, cudaMemcpyHostToDevice);
     cudaMemcpy(sym_pointState, lpointStates, sizeof(int) * DATASET_COUNT, cudaMemcpyHostToDevice);
     cudaMemcpy(sym_seedList, pointSeeds, sizeof(int) * THREAD_BLOCKS * MAX_SEEDS, cudaMemcpyHostToDevice);
 
     if (complete == THREAD_BLOCKS) 
         return false;
 
     return true;
 
 }
 
 void FinalizeClusters(int * states, int * clusterCount) {
     int16_t localCol[THREAD_BLOCKS][THREAD_BLOCKS];
 
     cudaMemcpy(localCol, sym_collision, sizeof(int16_t) * THREAD_BLOCKS * THREAD_BLOCKS, cudaMemcpyDeviceToHost);
     
     std::map<int,int> colMap;
     std::set<int> blockSet;
     for (int i = 0; i < THREAD_BLOCKS; i++) {
         colMap[i] = i;
         blockSet.insert(i);
     }
 
     std::set<int>::iterator it;
     do {
         it = blockSet.begin();
         int curBlock = *it; 
         std::set<int> expansionQueue; 
         std::set<int> finalQueue;
         finalQueue.insert(curBlock);
         expansionQueue.insert(curBlock);
         do {
             it = expansionQueue.begin();
             int expandBlock = *it;
             expansionQueue.erase(it);
             blockSet.erase(expandBlock);
             for (int x = 0; x < THREAD_BLOCKS; x++){
                 if (x == expandBlock)
                     continue;
                 if ((localCol[expandBlock][x] == 1 || localCol[x][expandBlock] == 1) 
                     && blockSet.find(x) != blockSet.end()) {
                     expansionQueue.insert(x);
                     finalQueue.insert(x);
                 }
             }
         } while (expansionQueue.empty() == 0);
 
         for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
             colMap[*it] = curBlock;
         }
     } while(blockSet.empty() == 0);
     std::vector<std::vector<int> > clusters(THREAD_BLOCKS, std::vector<int>());
     for (int i = 0; i < DATASET_COUNT; i++) {
         if (states[i] >= 0 && states[i] < THREAD_BLOCKS) {
             clusters[colMap[states[i]]].push_back(i);
         }
     }
 
     for (int i = 0; i < clusters.size(); i++) {
         if (clusters[i].size() == 0)
             continue;
         for(int x = 0; x < clusters[i].size(); x++) {
             states[clusters[i][x]] = *clusterCount + THREAD_BLOCKS + 1;
         }
         (*clusterCount)++;
     }
 
     cudaMemset(sym_collision, -1, sizeof(int16_t) * THREAD_BLOCKS * THREAD_BLOCKS);
     printf("Cluster Count:%d\n", *clusterCount);
 }
 
 
 // This is the DBSCAN kernel that runs in the GPU. 
 __global__ void DBSCAN(void){
     __shared__ int leave;
    __shared__ int nhood[MAX_SEEDS];
 
     __shared__ int nhoodCount;
 
     __shared__ float point[DIMENSION];

     __shared__ int pointID;

     int chainID = blockIdx.x;
 
     int seedLength = curSeedLength[chainID];
 
     nhoodCount = 0;
 
     if (seedLength == 0)
         return;
 
     seedLength = seedLength - 1;
     
     pointID = seedList[chainID][seedLength];
 
     point[0] = points[pointID][0];
     point[1] = points[pointID][1];
 
     leave = 0;
 
     __syncthreads();
 
     if (threadIdx.x == 0) {
         curSeedLength[chainID] = seedLength;
         if (pointState[pointID] != UNPROCESSED)
             leave = 1;
     }
     __syncthreads();
     if (leave == 1)
         return;
     
     __syncthreads();
 

     for(int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
         register float comp[2];
         comp[0] = points[i][0];
         comp[1] = points[i][1];
 
         register float delta = 0;
         delta = sqrtf(powf(comp[0]  - point[0], 2) + powf(comp[1] - point[1],2));
 
         if (delta <= eps) {
             register int h = atomicAdd(&nhoodCount,1);
             if (nhoodCount >= minpts) {
                 markAsCandidate(i, chainID);
             }
             else {
                 nhood[h] = i;
             }
         }
     }
 
     __syncthreads();
     if (threadIdx.x == 0 && nhoodCount > minpts) {
         nhoodCount = minpts;
     }
     __syncthreads();
 
     if (nhoodCount >= minpts) {
         pointState[pointID] = chainID;
         for (int i = threadIdx.x; i < nhoodCount; i = i + THREAD_COUNT) {
             markAsCandidate(nhood[i], chainID);
         }
     } else  {
         pointState[pointID] = NOISE;
     }
     __syncthreads();
     if (threadIdx.x == 0 && curSeedLength[chainID] >= MAX_SEEDS) {
         curSeedLength[chainID] = MAX_SEEDS - 1;
     }
 }
 
 // Mark a point as a member of a cluster. 
 __device__ void markAsCandidate(int pid, int chainID) {
     register int oldState = atomicCAS(&(pointState[pid]), 
                                       UNPROCESSED, chainID);    
     if (oldState == UNPROCESSED) {
         register int h = atomicAdd(&(curSeedLength[chainID]), 1);
         if (h < MAX_SEEDS) {
             seedList[chainID][h] = pid;
         } 
     } else if (oldState != NOISE && oldState != chainID && oldState < THREAD_BLOCKS) {
         if (oldState < chainID) {
             collisionMatrix[oldState][chainID] = 1;
         } else {
             collisionMatrix[chainID][oldState] = 1;
         }
     } else if (oldState == NOISE) {
         oldState = atomicCAS(&(pointState[pid]), NOISE, chainID);           
     }
 }

 int ImportDataset(char const *fname, float dataset[DATASET_COUNT][DIMENSION]) {
  FILE *fp = fopen(fname, "r");
  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }

  char buf[4096];
  unsigned long int cnt = 0;
  while (fgets(buf, 4096, fp) && cnt < DATASET_COUNT) {
    char *field = strtok(buf, ",");
    float tmp;
    unsigned int dim = 0;
    sscanf(field, "%f", &tmp);
    dataset[cnt][dim] = tmp;
    

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        dim++;
        float tmp;
        sscanf(field, "%f", &tmp);
        dataset[cnt][dim] = tmp;
        
      }
    }
    cnt++;
  }
  fclose(fp);
  return 0;
}