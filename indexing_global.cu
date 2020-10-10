#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#define DATASET_COUNT 1000
#define DIMENSION 2
#define TREE_LEVELS 3
#define RANGE 2
#define PARTITION 100
#define POINTS_SEARCHED 100
#define THREAD_BLOCKS 64
#define THREAD_COUNT 128

#define EPSILON 1.5

using namespace std;

struct __align__(8) IndexStructure {
  int level;
  double range[2];
  struct IndexStructure *buckets[PARTITION];
  int datas[POINTS_SEARCHED];
};

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
  * CUDA Data structures
  **************************************************************************
  */

__device__ int d_indexTreeMetaData[TREE_LEVELS][RANGE];
__device__ double d_dataset[DATASET_COUNT * DIMENSION];
__device__ int d_results[POINTS_SEARCHED];
__device__ int d_partition[DIMENSION];
__device__ double d_minPoints[DIMENSION];

int **sym_indexTreeMetaData;
double *sym_dataset;
int *sym_results;
int *sym_partition;
double *sym_minPoints;

void indexStructureInit(double *dataset);

__global__ void INDEXING_STRUCTURE(struct IndexStructure *indexRoot,
                               struct IndexStructure **indexBuckets, struct IndexStructure **currentIndexes, struct IndexStructure **indexesStack);

int ImportDataset(char const *fname, double *dataset);

__device__ void indexConstruction(int dimension,
                                  struct IndexStructure **indexBuckets);

__device__ void insertData(int id, struct IndexStructure *indexRoot, struct IndexStructure *currentIndex);

__device__ void searchPoints(int id, struct IndexStructure *indexRoot, struct IndexStructure *currentIndex, struct IndexStructure **indexesStack);

/**
//////////////////////////////////////////////////////////////////////////
**************************************************************************
* Main Function
**************************************************************************
//////////////////////////////////////////////////////////////////////////
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
  /**
 **************************************************************************
 * CUDA Memory allocation
 **************************************************************************
 */

  // Check if the GPU is functioning correctly
  gpuErrchk(cudaFree(0));

  struct IndexStructure *d_indexRoot;
  gpuErrchk(cudaMalloc((void **)&d_indexRoot, sizeof(struct IndexStructure)));

  /**
  **************************************************************************
  * CPU Memory allocation
  **************************************************************************
  */
  indexStructureInit(importedDataset);

  /**
 **************************************************************************
 * Copy data from cpu to gpu
 **************************************************************************
 */

  int partition[DIMENSION];
  gpuErrchk(cudaMemcpy(partition, sym_partition, sizeof(int) * DIMENSION,
                       cudaMemcpyDeviceToHost));

  int indexedStructureSize = 1;
  for (int i = 0; i < DIMENSION; i++) {
    indexedStructureSize *= partition[i];
  }

  for (int i = 0; i < DIMENSION - 1; i++) {
    indexedStructureSize += partition[i];
  }
  indexedStructureSize = indexedStructureSize + 1;

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
  cudaFree(d_currentIndexBucket);

  int TOTAL_THREADS = THREAD_BLOCKS * THREAD_COUNT;
  
  // Allocate memory for current indexed
  struct IndexStructure **d_currentIndexes, *d_currentIndex;

  gpuErrchk(cudaMalloc((void **)&d_currentIndexes,
                       sizeof(struct IndexStructure *) * TOTAL_THREADS));

  for (int i = 0; i < TOTAL_THREADS; i++) {
    gpuErrchk(cudaMalloc((void **)&d_currentIndex,
                         sizeof(struct IndexStructure)));
    gpuErrchk(cudaMemcpy(&d_currentIndexes[i], &d_currentIndex,
                         sizeof(struct IndexStructure *),
                         cudaMemcpyHostToDevice));
  }
  cudaFree(d_currentIndex);


  // Allocate memory for current indexes stack
  int indexBucketSize = RANGE;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }

  indexBucketSize = indexBucketSize * TOTAL_THREADS;
  
  struct IndexStructure **d_indexesStack, *d_currentIndexStack;

  gpuErrchk(cudaMalloc((void **)&d_indexesStack,
                       sizeof(struct IndexStructure *) * indexBucketSize));

  for (int i = 0; i < indexBucketSize; i++) {
    gpuErrchk(cudaMalloc((void **)&d_currentIndexStack,
                         sizeof(struct IndexStructure)));
    gpuErrchk(cudaMemcpy(&d_indexesStack[i], &d_currentIndexStack,
                         sizeof(struct IndexStructure *),
                         cudaMemcpyHostToDevice));
  }
  cudaFree(d_currentIndexStack);

  /**
**************************************************************************
* kernel Function...
**************************************************************************
*/

  INDEXING_STRUCTURE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(
      d_indexRoot, d_indexBuckets, d_currentIndexes, d_indexesStack);

  gpuErrchk(cudaDeviceSynchronize());

  /**
  **************************************************************************
  * Free CUDA memory allocations
  **************************************************************************
  */

  cudaFree(d_indexRoot);
  cudaFree(d_indexesStack);

  return 0;
}

void indexStructureInit(double *dataset) {
  gpuErrchk(cudaGetSymbolAddress((void **)&sym_dataset, d_dataset));
  gpuErrchk(cudaGetSymbolAddress((void **)&sym_results, d_results));

  gpuErrchk(cudaMemcpy(sym_dataset, dataset,
                       sizeof(double) * DATASET_COUNT * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemset(sym_results, -1, sizeof(int) * POINTS_SEARCHED));

  gpuErrchk(cudaGetSymbolAddress((void **)&sym_partition, d_partition));
  gpuErrchk(cudaGetSymbolAddress((void **)&sym_minPoints, d_minPoints));
  gpuErrchk(cudaGetSymbolAddress((void **)&sym_indexTreeMetaData,
                                 d_indexTreeMetaData));

  double maxPoints[DIMENSION];
  double minPoints[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    maxPoints[j] = 0;
    minPoints[j] = 999999999;
  }

  for (int i = 0; i < DATASET_COUNT; i++) {
    for (int j = 0; j < DIMENSION; j++) {
      if (dataset[i * DIMENSION + j] > maxPoints[j]) {
        maxPoints[j] = dataset[i * DIMENSION + j];
      }
      if (dataset[i * DIMENSION + j] < minPoints[j]) {
        minPoints[j] = dataset[i * DIMENSION + j];
      }
    }
  }

  int *partition = (int *)malloc(sizeof(int) * DIMENSION);

  for (int i = 0; i < DIMENSION; i++) {
    partition[i] = 0;
    double curr = minPoints[i];
    while (curr < maxPoints[i]) {
      partition[i]++;
      curr += EPSILON;
    }
  }

  int treeLevelPartition[TREE_LEVELS] = {1};

  for (int i = 0; i < DIMENSION; i++) {
    treeLevelPartition[i + 1] = partition[i];
  }

  int childItems[TREE_LEVELS];
  int startEndIndexes[TREE_LEVELS][2];

  int mulx = 1;
  for (int k = 0; k < TREE_LEVELS; k++) {
    mulx *= treeLevelPartition[k];
    childItems[k] = mulx;
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    if (i == 0) {
      startEndIndexes[i][0] = 0;
      startEndIndexes[i][1] = 1;
      continue;
    }
    startEndIndexes[i][0] = startEndIndexes[i - 1][1];
    startEndIndexes[i][1] = startEndIndexes[i][0];
    for (int k = 0; k < childItems[i - 1]; k++) {
      startEndIndexes[i][1] += treeLevelPartition[i];
    }
  }

  gpuErrchk(cudaMemcpy(sym_partition, partition, sizeof(int) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(sym_minPoints, minPoints, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(sym_indexTreeMetaData, startEndIndexes,
                       sizeof(int) * TREE_LEVELS * RANGE,
                       cudaMemcpyHostToDevice));
}

__global__ void INDEXING_STRUCTURE(struct IndexStructure *indexRoot,
                               struct IndexStructure **indexBuckets, struct IndexStructure **currentIndexes, struct IndexStructure **indexesStack) {

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    indexBuckets[0] = indexRoot;
  }
  __syncthreads();


  for(int i = 0; i <= DIMENSION; i++) {
    indexConstruction(i, indexBuckets);
    __syncthreads();
  }

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  
  for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
    insertData(i, indexRoot, currentIndexes[threadId]);
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    searchPoints(999, indexRoot, currentIndexes[threadId], indexesStack);
    __syncthreads();
    printf("Searching for point 0: ");
    for (int i = 0; i < POINTS_SEARCHED; i++) {
      printf("%d ", d_results[i]);
    }
    printf("\n======================\n");
  }
  __syncthreads();
}

__device__ void indexConstruction(int dimension,
                                  struct IndexStructure **indexBuckets) {
  if (dimension > DIMENSION) return;
  for (int k = blockIdx.x + d_indexTreeMetaData[dimension][0];
       k < d_indexTreeMetaData[dimension][1]; k = k + THREAD_BLOCKS) {
    
    if (dimension >= DIMENSION && threadIdx.x == 0) {
      for (int i = 0; i < POINTS_SEARCHED; i++) {
        indexBuckets[k]->datas[i] = -1;
      }
      continue;
    }

    for (int i = threadIdx.x; i < d_partition[dimension]; i = i + THREAD_BLOCKS) {
      int currentBucketIndex =
          d_indexTreeMetaData[dimension][1] + i +
          (k - d_indexTreeMetaData[dimension][0]) * d_partition[dimension];

      indexBuckets[k]->buckets[i] = indexBuckets[currentBucketIndex];
      indexBuckets[k]->level = dimension;

      double leftPoint = d_minPoints[dimension] + i * EPSILON;
      double rightPoint = leftPoint + EPSILON;

      indexBuckets[k]->buckets[i]->range[0] = leftPoint;
      indexBuckets[k]->buckets[i]->range[1] = rightPoint;
    }
  }
}

__device__ void insertData(int id, struct IndexStructure *indexRoot, struct IndexStructure *currentIndex) {
  register float data[DIMENSION];
  for (int j = 0; j < DIMENSION; j++) {
    data[j] = d_dataset[id * DIMENSION + j];
  }

  currentIndex = indexRoot;
  bool found = false;

  while (!found) {
    int dimension = currentIndex->level;
    for (int k = 0; k < d_partition[dimension]; k++) {

      register float comparingData = data[dimension];
      register float leftRange = currentIndex->buckets[k]->range[0];
      register float rightRange = currentIndex->buckets[k]->range[1];

      if (comparingData >= leftRange && comparingData <= rightRange) {
        if (dimension == DIMENSION - 1) {
          for (int i = 0; i < POINTS_SEARCHED; i++) {
            register int changedState =
                atomicCAS(&currentIndex->buckets[k]->datas[i], -1, id);
            if (changedState == -1 || changedState == id) {
              break;
            }
          }
          found = true;
          break;
        }
        currentIndex = currentIndex->buckets[k];
        break;
      }
    }
  }
}

__device__ void searchPoints(int id, struct IndexStructure *indexRoot, struct IndexStructure *currentIndex, struct IndexStructure **indexesStack) {
  double data[DIMENSION];
  for (int i = 0; i < DIMENSION; i++) {
    data[i] = d_dataset[id * DIMENSION + i];
  }

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;

  int indexBucketSize = RANGE;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }

  int currentIndexSize = threadId*indexBucketSize;
  indexesStack[currentIndexSize++] = indexRoot;

  int resultsCount = 0;

  while (currentIndexSize > threadId*indexBucketSize) {
    
    currentIndex = indexesStack[--currentIndexSize];

    int dimension = currentIndex->level;

    for (int k = 0; k < d_partition[dimension]; k++) {
      float comparingData = (float)data[dimension];
      float leftRange = (float)currentIndex->buckets[k]->range[0];
      float rightRange = (float)currentIndex->buckets[k]->range[1];

      if (comparingData >= leftRange && comparingData <= rightRange) {
        if (dimension == DIMENSION - 1) {
          for (int i = 0; i < POINTS_SEARCHED; i++) {
            if (currentIndex->buckets[k]->datas[i] != -1) {
              d_results[resultsCount++] = currentIndex->buckets[k]->datas[i];
              continue;
            }
            break;
          }

          if (k > 0) {
            for (int i = 0; i < POINTS_SEARCHED; i++) {
              if (currentIndex->buckets[k - 1]->datas[i] != -1) {
                d_results[resultsCount++] =
                    currentIndex->buckets[k - 1]->datas[i];
                continue;
              }
              break;
            }
          }
          if (k < d_partition[dimension] - 1) {
            for (int i = 0; i < POINTS_SEARCHED; i++) {
              if (currentIndex->buckets[k + 1]->datas[i] != -1) {
                d_results[resultsCount++] =
                    currentIndex->buckets[k + 1]->datas[i];
                continue;
              }
              break;
            }
          }
          break;
        }
        indexesStack[currentIndexSize++] = currentIndex->buckets[k];
        if (k > 0) {
          indexesStack[currentIndexSize++] = currentIndex->buckets[k - 1];
        }
        if (k < d_partition[dimension] - 1) {
          indexesStack[currentIndexSize++] = currentIndex->buckets[k + 1];
        }
        break;
      }
    }
  }
}

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