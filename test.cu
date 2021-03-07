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

#define THREAD_BLOCKS 12
#define THREAD_COUNT 12

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

__global__ void COLLISION_DETECTION(int *collisionMatrix);

int main() {
  int collisionMatrix[THREAD_BLOCKS][THREAD_BLOCKS] = {
      {1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
      {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0},
      {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0},
      {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0},
      {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0}};

  int colMap[THREAD_BLOCKS];
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
      expansionQueue.erase(expandBlock);
      blockSet.erase(expandBlock);
      for (int x = 0; x < THREAD_BLOCKS; x++) {
        if (x == expandBlock) continue;
        if ((collisionMatrix[expandBlock][x] == 1 ||
             collisionMatrix[x][expandBlock]) &&
            blockSet.find(x) != blockSet.end()) {
          expansionQueue.insert(x);
          finalQueue.insert(x);
        }
      }
    } while (expansionQueue.empty() == 0);

    for (it = finalQueue.begin(); it != finalQueue.end(); ++it) {
      colMap[*it] = curBlock;
    }
  } while (blockSet.empty() == 0);

  for (int i = 0; i < THREAD_BLOCKS; i++) {
    cout << i << ": " << colMap[i] << endl;
  }

  cout << "############################" << endl;

  int *d_collisionMatrix;
  gpuErrchk(cudaMalloc((void **)&d_collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS));

  gpuErrchk(cudaMemcpy(d_collisionMatrix, collisionMatrix,
                       sizeof(int) * THREAD_BLOCKS * THREAD_BLOCKS,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaDeviceSynchronize());
  COLLISION_DETECTION<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_collisionMatrix);
  gpuErrchk(cudaDeviceSynchronize());
  return 0;
}

__global__ void COLLISION_DETECTION(int *collisionMatrix) {
  if (blockIdx.x == 0) {
    __shared__ int colMap[THREAD_BLOCKS];
    __shared__ int blockSet[THREAD_BLOCKS];
    __shared__ int blocksetCount;

    if (threadIdx.x == 0) {
      for (int i = 0; i < THREAD_BLOCKS; i++) {
        colMap[i] = i;
        blockSet[i] = i;
      }
      blocksetCount = THREAD_BLOCKS;
    }
    __syncthreads();

    __shared__ int curBlock;
    __shared__ int expansionQueue[THREAD_BLOCKS];
    __shared__ int finalQueue[THREAD_BLOCKS];
    __shared__ int expansionQueueCount;
    __shared__ int finalQueueCount;
    __shared__ int expandBlock;

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
          int oldExpansionQueueCount = atomicSub(&expansionQueueCount, 1);
          expandBlock = expansionQueue[oldExpansionQueueCount - 1];
        }
        __syncthreads();

        thrust::remove(thrust::device, expansionQueue,
                       expansionQueue + THREAD_BLOCKS, expandBlock);
        thrust::remove(thrust::device, blockSet, blockSet + THREAD_BLOCKS,
                       expandBlock);

        if (threadIdx.x == 0) {
          atomicSub(&blocksetCount, 1);
        }
        __syncthreads();

        for (int x = threadIdx.x; x < THREAD_BLOCKS; x = x + THREAD_COUNT) {
          if (x == expandBlock) continue;
          if ((collisionMatrix[expandBlock * THREAD_BLOCKS + x] == 1 ||
               collisionMatrix[x * THREAD_BLOCKS + expandBlock]) &&
              thrust::find(thrust::device, blockSet, blockSet + THREAD_BLOCKS,
                           x) != blockSet + THREAD_BLOCKS) {
            if (thrust::find(thrust::device, expansionQueue,
                             expansionQueue + THREAD_BLOCKS,
                             x) == expansionQueue + THREAD_BLOCKS) {
              int oldExpansionQueueCount = atomicAdd(&expansionQueueCount, 1);
              expansionQueue[oldExpansionQueueCount] = x;
            }

            if (thrust::find(thrust::device, finalQueue,
                             finalQueue + THREAD_BLOCKS,
                             x) == finalQueue + THREAD_BLOCKS) {
              int oldFinalQueueCount = atomicAdd(&finalQueueCount, 1);
              finalQueue[oldFinalQueueCount] = x;
            }
          }
        }
      };

      for (int c = threadIdx.x; c < finalQueueCount; c = c + THREAD_COUNT) {
        colMap[finalQueue[c]] = curBlock;
      }
      __syncthreads();
    };
    if (threadIdx.x == 0) {
      for (int i = 0; i < THREAD_BLOCKS; i++) {
        printf("%d -> %d\n", i, colMap[i]);
      }
    }
  }
}