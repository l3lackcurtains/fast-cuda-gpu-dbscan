#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#define DATASET_COUNT 1000
#define DIMENSION 2
#define TREE_LEVELS DIMENSION + 1
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


__device__ double d_dataset[DATASET_COUNT*DIMENSION];
__device__ int d_results[POINTS_SEARCHED];
__device__ int d_partition[DIMENSION];
__device__ double d_minPoints[DIMENSION];

double * sym_dataset;
int * sym_results;
int * sym_partition;
double * sym_minPoints;

__global__ void Indexingkernel(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets, int * indexBucketsLength);

int ImportDataset(char const *fname, double *dataset);

__device__ void indexConstruction(int dimension, struct IndexStructure **indexBuckets);

__device__ void indexConstruction2D(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets);

__device__
void insertData(int id, struct IndexStructure *indexRoot);

__device__
void searchPoints(int id, struct IndexStructure *indexRoot);

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

   gpuErrchk(cudaGetSymbolAddress((void **)&sym_dataset, d_dataset));
   gpuErrchk(cudaGetSymbolAddress((void **)&sym_partition, d_partition));
   gpuErrchk(cudaGetSymbolAddress((void **)&sym_results, d_results));
   gpuErrchk(cudaGetSymbolAddress((void **)&sym_minPoints, d_minPoints)); 

    struct IndexStructure *d_indexRoot;
    gpuErrchk(cudaMalloc((void **)&d_indexRoot, sizeof(struct IndexStructure)));

   /**
   **************************************************************************
   * CPU Memory allocation
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
            curr += EPSILON;
        }
    }

    /**
   **************************************************************************
   * Copy data from cpu to gpu
   **************************************************************************
   */

   gpuErrchk(cudaMemcpy(sym_dataset, importedDataset,
    sizeof(double) * DATASET_COUNT * DIMENSION,
    cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(sym_minPoints, minPoints,
        sizeof(double) * DIMENSION,
        cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(sym_partition, partition,
            sizeof(int) * DIMENSION,
            cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(sym_results, -1, sizeof(int) * POINTS_SEARCHED));

    int indexedStructureSize = 1;
    for(int i = 0; i < DIMENSION; i++) {
        indexedStructureSize *= partition[i];
    }

    for(int i = 0; i < DIMENSION - 1; i++) {
        indexedStructureSize += partition[i];
    }
    indexedStructureSize = indexedStructureSize + 1;

    int * d_indexBucketsLength, * d_dataNodesLength;
    gpuErrchk(cudaMalloc((void **)&d_indexBucketsLength,sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&d_dataNodesLength, sizeof(int)));

    gpuErrchk(cudaMemset(d_indexBucketsLength, 0, sizeof(int)));
    gpuErrchk(cudaMemset(d_dataNodesLength, 0, sizeof(int)));
    
    
    // Allocate memory for index buckets
    struct IndexStructure **d_indexBuckets, *d_currentIndexBucket;

    gpuErrchk(cudaMalloc((void **)&d_indexBuckets,sizeof(struct IndexStructure*) * indexedStructureSize));

    for(int i = 0; i < indexedStructureSize; i++) {
        gpuErrchk(cudaMalloc((void **)&d_currentIndexBucket, sizeof(struct IndexStructure)));
        gpuErrchk(cudaMemcpy(&d_indexBuckets[i], &d_currentIndexBucket, sizeof(struct IndexStructure*), cudaMemcpyHostToDevice));
        
    }
    cudaFree(d_currentIndexBucket);

     /**
   **************************************************************************
   * kernel Function...
   **************************************************************************
   */

   
   cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
   Indexingkernel<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>( d_indexRoot, d_indexBuckets, d_indexBucketsLength);

   gpuErrchk(cudaDeviceSynchronize());

   /**
   **************************************************************************
   * Free CUDA memory allocations
   **************************************************************************
   */

    cudaFree(d_indexRoot);
    
    return 0;
}

__global__ void Indexingkernel(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets, int * indexBucketsLength) {

    __shared__ int blockID;

    if (threadIdx.x == 0) {
        blockID = blockIdx.x;
        indexBuckets[0] = indexRoot;
    }
    __syncthreads();

    indexConstruction(blockID, indexBuckets);
    
    __syncthreads();
    
    for (int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
        insertData(i, indexRoot);
    }
    __syncthreads();
    

    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
            searchPoints(0, indexRoot);
            __syncthreads();
            printf("Searching for point 0: ");
            for(int i = 0; i < POINTS_SEARCHED; i++) {
                printf("%d ", d_results[i]);
            }
            printf("\n======================\n");
    }
    __syncthreads();
}

__device__ void indexConstruction(int dimension, struct IndexStructure **indexBuckets) {

    if(dimension > DIMENSION) return;

    int partition[TREE_LEVELS] = {1};

    for(int i = 0; i < DIMENSION; i++) {
        partition[i+1] = d_partition[i];
    }

    int childItems[TREE_LEVELS];
    int startEndIndexes[TREE_LEVELS][2];

    int mulx = 1;
    for(int k = 0; k < TREE_LEVELS; k++) {
        mulx *= partition[k];
        childItems[k] = mulx;
    }

    for(int i = 0; i < TREE_LEVELS; i++) {
        if(i == 0) {
            startEndIndexes[i][0] = 0;
            startEndIndexes[i][1] = 1;
            continue;
        }
        startEndIndexes[i][0] = startEndIndexes[i-1][1];
        startEndIndexes[i][1] =  startEndIndexes[i][0];
        for(int k = 0; k < childItems[i-1]; k++) {
            startEndIndexes[i][1] += partition[i];
        }

    }

    for(int k = threadIdx.x + startEndIndexes[dimension][0]; k < startEndIndexes[dimension][1]; k = k + THREAD_COUNT) {
        if(dimension < DIMENSION) {
            for(int i = 0; i < d_partition[dimension]; i++) {
                
                int currentBucketIndex = startEndIndexes[dimension][1] + i + (k - startEndIndexes[dimension][0])*d_partition[dimension];

                indexBuckets[k]->buckets[i] = indexBuckets[currentBucketIndex];
                indexBuckets[k]->level = dimension;
            
                double leftPoint = d_minPoints[dimension] + i* EPSILON;
                double rightPoint = leftPoint + EPSILON;

                indexBuckets[k]->buckets[i]->range[0] = leftPoint;
                indexBuckets[k]->buckets[i]->range[1] = rightPoint;
            }
        } else {
            for(int i = 0; i < POINTS_SEARCHED; i++) {
                indexBuckets[k]->datas[i] = -1;
            }
        }
    }
}

__device__
void insertData(int id, struct IndexStructure *indexRoot) {

    register float data[DIMENSION];
    for (int j = 0; j < DIMENSION; j++) {
        data[j] = d_dataset[id * DIMENSION + j];
    }

    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

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
                            atomicCAS(&currentIndex->buckets[k]->datas[i],
                                      -1, id);
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


__device__
void searchPoints(int id, struct IndexStructure *indexRoot) {

    double data[DIMENSION];
    for(int i = 0; i < DIMENSION; i++) {
        data[i] = d_dataset[id*DIMENSION + i];
    }

    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    // Size of data Node and index
    int indexedStructureSize = 2;
    for(int i = 0; i < DIMENSION; i++) {
      indexedStructureSize *= 3;
    }

    // Current Index
    struct IndexStructure ** currentIndexes = (struct IndexStructure**) malloc(sizeof(struct IndexStructure*) * indexedStructureSize);

    int currentIndexSize = 0;
    currentIndexes[currentIndexSize++] = indexRoot;

    // Selected data Node 
    struct dataNode ** selectedDataNodes = (struct dataNode **) malloc(sizeof(struct dataNode *) * indexedStructureSize);

    int resultsCount = 0;

    while (currentIndexSize > 0) {

        currentIndex = currentIndexes[--currentIndexSize];

        int dimension = currentIndex->level;

        for (int k = 0; k < d_partition[dimension]; k++) {

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentIndex->buckets[k]->range[0];
            float rightRange = (float)currentIndex->buckets[k]->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {

                  
                    for(int i = 0; i < POINTS_SEARCHED; i++) {
                        if(currentIndex->buckets[k]->datas[i] != -1) {
                            d_results[resultsCount++] = currentIndex->buckets[k]->datas[i];
                            continue;
                        }
                        break;
                    }

                    if (k > 0) {
                        for(int i = 0; i < POINTS_SEARCHED; i++) {
                            if(currentIndex->buckets[k-1]->datas[i] != -1) {
                                d_results[resultsCount++] = currentIndex->buckets[k-1]->datas[i];
                            continue;
                            }
                            break;
                        }
                    }
                    if (k < d_partition[dimension] - 1) {
                        for(int i = 0; i < POINTS_SEARCHED; i++) {
                            if(currentIndex->buckets[k+1]->datas[i] != -1) {
                                d_results[resultsCount++] = currentIndex->buckets[k+1]->datas[i];
                            continue;
                            }
                            break;
                        }
                    }
                    break;
                }
                currentIndexes[currentIndexSize++] = currentIndex->buckets[k];
                if (k > 0) {
                    currentIndexes[currentIndexSize++] = currentIndex->buckets[k - 1];
                }
                if (k < d_partition[dimension] - 1) {
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