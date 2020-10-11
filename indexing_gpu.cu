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

#define EPS 1.5

using namespace std;

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

  struct __align__(8) IndexStructure {
    int level;
    double range[2];
    struct IndexStructure *buckets[PARTITION];
    int datas[POINTS_SEARCHED];
  };

__global__ void INDEXING_STRUCTURE(double * dataset, int * indexTreeMetaData, double * minPoints, int * partition, int * results, struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets);

int ImportDataset(char const *fname, double *dataset);

__device__ void indexConstruction(int dimension, int * indexTreeMetaData, int * partition, double * minPoints, struct IndexStructure **indexBuckets);

__device__ void insertData(int id, double * dataset, int * partition, struct IndexStructure *indexRoot);

__device__ void searchPoints(int id, int chainID, double *dataset, int * partition, int * results, struct IndexStructure *indexRoot);

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

    // Check if the GPU is functioning correctly
    gpuErrchk(cudaFree(0));

  /**
 **************************************************************************
 * CUDA Memory allocation
 **************************************************************************
 */

  double *d_dataset;
  int *d_indexTreeMetaData;
  int *d_results;
  int *d_partition;
  double *d_minPoints;

  gpuErrchk(cudaMalloc((void **)&d_dataset, sizeof(double) * DATASET_COUNT * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_indexTreeMetaData, sizeof(int) * TREE_LEVELS * RANGE));

  gpuErrchk(cudaMalloc((void **)&d_results, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

  gpuErrchk(cudaMalloc((void **)&d_partition, sizeof(int) * DIMENSION));

  gpuErrchk(cudaMalloc((void **)&d_minPoints, sizeof(double) * DIMENSION));


  gpuErrchk(cudaMemcpy(d_dataset, importedDataset,
    sizeof(double) * DATASET_COUNT * DIMENSION,
    cudaMemcpyHostToDevice));

  struct IndexStructure *d_indexRoot;
  gpuErrchk(cudaMalloc((void **)&d_indexRoot, sizeof(struct IndexStructure)));

  gpuErrchk(cudaMemset(d_results, -1, sizeof(int) * THREAD_BLOCKS * POINTS_SEARCHED));

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

  int *partition = (int *)malloc(sizeof(int) * DIMENSION);

  for (int i = 0; i < DIMENSION; i++) {
    partition[i] = 0;
    double curr = minPoints[i];
    while (curr < maxPoints[i]) {
      partition[i]++;
      curr += EPS;
    }
  }

  int treeLevelPartition[TREE_LEVELS] = {1};

  for (int i = 0; i < DIMENSION; i++) {
    treeLevelPartition[i + 1] = partition[i];
  }

  int childItems[TREE_LEVELS];
  int startEndIndexes[TREE_LEVELS*RANGE];

  int mulx = 1;
  for (int k = 0; k < TREE_LEVELS; k++) {
    mulx *= treeLevelPartition[k];
    childItems[k] = mulx;
  }

  for (int i = 0; i < TREE_LEVELS; i++) {
    if (i == 0) {
      startEndIndexes[i*RANGE + 0] = 0;
      startEndIndexes[i*RANGE + 1] = 1;
      continue;
    }
    startEndIndexes[i*RANGE + 0] = startEndIndexes[((i - 1)*RANGE) + 1];
    startEndIndexes[i*RANGE + 1] = startEndIndexes[i*RANGE + 0];
    for (int k = 0; k < childItems[i - 1]; k++) {
      startEndIndexes[i*RANGE + 1] += treeLevelPartition[i];
    }
  }

  gpuErrchk(cudaMemcpy(d_partition, partition, sizeof(int) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_minPoints, minPoints, sizeof(double) * DIMENSION,
                       cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy(d_indexTreeMetaData, startEndIndexes,
                       sizeof(int) * TREE_LEVELS * RANGE,
                       cudaMemcpyHostToDevice));

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

  /**
**************************************************************************
* kernel Function...
**************************************************************************
*/

  INDEXING_STRUCTURE<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>(d_dataset, d_indexTreeMetaData, d_minPoints, d_partition, d_results, d_indexRoot, d_indexBuckets);

  gpuErrchk(cudaDeviceSynchronize());

  /**
  **************************************************************************
  * Free CUDA memory allocations
  **************************************************************************
  */

  cudaFree(d_indexRoot);

  return 0;
}

__global__ void INDEXING_STRUCTURE(double * dataset, int * indexTreeMetaData, double * minPoints, int * partition, int * results, struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets) {


  __shared__ int chainID;

  if (threadIdx.x == 0) {
    chainID = blockIdx.x;
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    indexBuckets[0] = indexRoot;
  }
  __syncthreads();


  for(int i = 0; i <= DIMENSION; i++) {
    indexConstruction(i, indexTreeMetaData, partition, minPoints, indexBuckets);
    __syncthreads();
  }

  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  for (int i = threadId; i < DATASET_COUNT; i = i + THREAD_COUNT*THREAD_BLOCKS) {
    insertData(i, dataset, partition, indexRoot);
  }

  
  __syncthreads();

  if (threadIdx.x == 0 && chainID == 0) {

    searchPoints(199, 0, dataset, partition, results, indexRoot);
    __syncthreads();

    printf("Searching for point %d:\n", chainID);

    for (int i = 0; i < POINTS_SEARCHED; i++) {
      printf("%d ", results[chainID * POINTS_SEARCHED + i]);
    }
    printf("\n======================\n");
  }
  __syncthreads();
}

__device__ void indexConstruction(int dimension, int * indexTreeMetaData, int * partition, double * minPoints, struct IndexStructure **indexBuckets) {


  if (dimension > DIMENSION) return;

  if(blockIdx.x == 0 && threadIdx.x == 0) {
    printf("%d %d\n", indexTreeMetaData[dimension*RANGE + 0], indexTreeMetaData[dimension*RANGE + 1]);
  }

  for (int k = blockIdx.x + indexTreeMetaData[dimension*RANGE + 0];
       k < indexTreeMetaData[dimension*RANGE + 1]; k = k + THREAD_BLOCKS) {
    
    if (dimension >= DIMENSION && threadIdx.x == 0) {
      for (int i = 0; i < POINTS_SEARCHED; i++) {
        indexBuckets[k]->datas[i] = -1;
      }
      continue;
    }

    for (int i = threadIdx.x; i < partition[dimension]; i = i + THREAD_BLOCKS) {
      int currentBucketIndex =
          indexTreeMetaData[dimension*RANGE + 1] + i +
          (k - indexTreeMetaData[dimension * RANGE + 0]) * partition[dimension];

      indexBuckets[k]->buckets[i] = indexBuckets[currentBucketIndex];
      indexBuckets[k]->level = dimension;

      double leftPoint = minPoints[dimension] + i * EPS;
      double rightPoint = leftPoint + EPS;

      indexBuckets[k]->buckets[i]->range[0] = leftPoint;
      indexBuckets[k]->buckets[i]->range[1] = rightPoint;
    }
  }

}

__device__ void insertData(int id, double * dataset, int * partition, struct IndexStructure *indexRoot) {


  register float data[DIMENSION];

  for (int j = 0; j < DIMENSION; j++) {
    data[j] = dataset[id * DIMENSION + j];
  }

  struct IndexStructure *currentIndex =
      (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

  currentIndex = indexRoot;
  bool found = false;

  while (!found) {
    int dimension = currentIndex->level;
    for (int k = 0; k < partition[dimension]; k++) {
      float comparingData = (float) data[dimension];
      float leftRange = (float) currentIndex->buckets[k]->range[0];
      float rightRange = (float) currentIndex->buckets[k]->range[1];

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

__device__ void searchPoints(int id, int chainID, double *dataset, int * partition, int * results, struct IndexStructure *indexRoot) {
  
  for (int i = 0; i < POINTS_SEARCHED; i++) {
    results[chainID * POINTS_SEARCHED + i] = -1;
  }

  double data[DIMENSION];
  for (int i = 0; i < DIMENSION; i++) {
    data[i] = dataset[id * DIMENSION + i];
  }

  struct IndexStructure *currentIndex =
      (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
  int indexBucketSize = 1;
  for (int i = 0; i < DIMENSION; i++) {
    indexBucketSize *= 3;
  }

  // Current Index
  struct IndexStructure **currentIndexes = (struct IndexStructure **)malloc(
      sizeof(struct IndexStructure *) * indexBucketSize);

  int currentIndexSize = 0;
  currentIndexes[currentIndexSize++] = indexRoot;

  int resultsCount = 0;

  while (currentIndexSize > 0) {
    
    currentIndex = currentIndexes[--currentIndexSize];

    int dimension = currentIndex->level;

    for (int k = 0; k < partition[dimension]; k++) {     

      float comparingData = (float)data[dimension];
      float leftRange = (float)currentIndex->buckets[k]->range[0];
      float rightRange = (float)currentIndex->buckets[k]->range[1];

      if (comparingData >= leftRange && comparingData <= rightRange) {

        
        if (dimension == DIMENSION - 1) {
         
          for (int i = 0; i < POINTS_SEARCHED; i++) {

            if (currentIndex->buckets[k]->datas[i] == -1) {
              break;
            }
            results[chainID * POINTS_SEARCHED + resultsCount] = currentIndex->buckets[k]->datas[i];
              resultsCount++;
          }

          if (k > 0 && currentIndex->buckets[k - 1] != NULL) {
            for (int i = 0; i < POINTS_SEARCHED; i++) {
              if (currentIndex->buckets[k - 1]->datas[i] == -1) {
                break;
              }
              results[chainID * POINTS_SEARCHED + resultsCount] =
                    currentIndex->buckets[k - 1]->datas[i];
                    resultsCount++;
            }
          }
          if (k < partition[dimension] - 1 && currentIndex->buckets[k + 1] != NULL) {
            for (int i = 0; i < POINTS_SEARCHED; i++) {
              if (currentIndex->buckets[k + 1]->datas[i] == -1) {
                break;
              }
              results[chainID * POINTS_SEARCHED + resultsCount] =
                    currentIndex->buckets[k + 1]->datas[i];
                    resultsCount++;
            }
          }
          break;
        }
        currentIndexes[currentIndexSize++] = currentIndex->buckets[k];
        if (k > 0) {
          currentIndexes[currentIndexSize++] = currentIndex->buckets[k - 1];
        }
        if (k < partition[dimension] - 1) {
          currentIndexes[currentIndexSize++] = currentIndex->buckets[k + 1];
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