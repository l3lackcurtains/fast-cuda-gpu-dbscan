#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#define DATASET_COUNT 1000
#define DIMENSION 2
#define PARTITION 100
#define POINTS_SEARCHED 100
#define THREAD_BLOCKS 64
#define THREAD_COUNT 128

#define EPSILON 1.5

using namespace std;

struct __align__(8) dataNode {
    int id;
    struct dataNode *child;
};

struct __align__(8) IndexStructure {
    int level;
    double range[2];
    struct IndexStructure *buckets[PARTITION];
    struct dataNode *dataRoot;
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

__global__ void Indexingkernel(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets, struct dataNode **dataNodes, int * indexBucketsLength, int * dataNodesLength);

int ImportDataset(char const *fname, double *dataset);

__device__ void indexConstruction(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets, int * indexBucketsLength, struct dataNode **dataNodes, int * dataNodesLength);

__device__ void indexConstruction2D(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets);

__device__
void insertData(int id, struct IndexStructure *indexRoot, struct dataNode **dataNodes, int * dataNodesLength);

__device__
void searchPoints(double *data, struct IndexStructure *indexRoot);

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

    int indexedStructureSize = 2;
    for(int i = 0; i < DIMENSION; i++) {
        indexedStructureSize *= partition[i];
    }


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

    // Allocate memory for data Nodes
    struct dataNode **d_dataNodes, *d_currentdataNode;

    gpuErrchk(cudaMalloc((void **)&d_dataNodes,sizeof(struct dataNode*) * DATASET_COUNT));

    for(int i = 0; i < DATASET_COUNT; i++) {
        gpuErrchk(cudaMalloc((void **)&d_currentdataNode,sizeof(struct dataNode*)));
        gpuErrchk(cudaMemcpy(&d_dataNodes[i], &d_currentdataNode, sizeof(struct dataNode*), cudaMemcpyHostToDevice));
    }
    cudaFree(d_currentdataNode);

     /**
   **************************************************************************
   * kernel Function...
   **************************************************************************
   */

   
   cudaThreadSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
   Indexingkernel<<<dim3(THREAD_BLOCKS, 1), dim3(THREAD_COUNT, 1)>>>( d_indexRoot, d_indexBuckets, d_dataNodes, d_indexBucketsLength, d_dataNodesLength);

   gpuErrchk(cudaDeviceSynchronize());

   /**
   **************************************************************************
   * Free CUDA memory allocations
   **************************************************************************
   */

    cudaFree(d_indexRoot);
    
    return 0;
}

__global__ void Indexingkernel(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets, struct dataNode **dataNodes, int * indexBucketsLength, int * dataNodesLength) {

    for(int i = threadIdx.x; i < DATASET_COUNT; i = i + THREAD_COUNT) {
        dataNodes[i]->id = i;
        
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        indexConstruction2D(indexRoot, indexBuckets);
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
    for(int i = 0; i < DATASET_COUNT; i++) {
        insertData(i, indexRoot, dataNodes, dataNodesLength);
    }
}
    __syncthreads();

    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        
        double data[DIMENSION];
        data[0] = d_dataset[0];
        data[1] = d_dataset[1];
        searchPoints(data, indexRoot);

        __syncthreads();

        for(int i = 0; i < POINTS_SEARCHED; i++) {
            printf("%d ", d_results[i]);
        }
        printf("\n");
    }

    __syncthreads();
    
    

}


__device__ void indexConstruction(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets, int * indexBucketsLength, struct dataNode **dataNodes, int * dataNodesLength) {

    int indexedStructureSize = 2;
    for(int i = 0; i < DIMENSION; i++) {
      indexedStructureSize *= d_partition[i];
    }

    struct IndexStructure ** indexedStructures = (struct IndexStructure**) malloc(sizeof(struct IndexStructure*) * indexedStructureSize);

    int indexedStructureSizeCount = 0;
    indexedStructures[indexedStructureSizeCount++] = indexRoot;

    for (int j = 0; j < DIMENSION; j++) {

        struct IndexStructure ** childIndexedStructures = (struct IndexStructure**) malloc(sizeof(struct IndexStructure*) * indexedStructureSize);
        int childIndexedStructureSizeCount = 0;

        while (indexedStructureSizeCount > 0) {

            struct IndexStructure *currentIndex = indexedStructures[--indexedStructureSizeCount];
            currentIndex->level = j;

            double rightPoint = d_minPoints[j] + d_partition[j] * EPSILON;

            for (int k = d_partition[j] - 1; k >= 0; k--) {
                register int currentBucketCount = atomicAdd(indexBucketsLength, 1);

                struct IndexStructure *currentBucket = indexBuckets[currentBucketCount];

                currentBucket->range[1] = rightPoint;
                rightPoint = rightPoint - EPSILON;
                currentBucket->range[0] = rightPoint;

                if(j == DIMENSION -1) {
                    register int currentDataNodeCount = atomicAdd(dataNodesLength, 1);
                    currentBucket->dataRoot = dataNodes[currentDataNodeCount];
                    currentBucket->dataRoot->id = -1;
                }

                currentIndex->buckets[k] = currentBucket;
                if (j < DIMENSION - 1) {
                    childIndexedStructures[childIndexedStructureSizeCount++] = currentIndex->buckets[k];
                }
            }
        }

        while (childIndexedStructureSizeCount > 0) {
          indexedStructures[indexedStructureSizeCount++] = childIndexedStructures[--childIndexedStructureSizeCount];

          free(childIndexedStructures[childIndexedStructureSizeCount]);
        }

        free(childIndexedStructures);
    }

    for(int i = 0; i < indexedStructureSize; i++) {
        free(indexedStructures[i]);
    }
    free(indexedStructures);
}

__device__ void indexConstruction2D(struct IndexStructure *indexRoot, struct IndexStructure **indexBuckets) {

    int currentLevel = 0;
    indexRoot->level = currentLevel;

    int currentBucketCount = 0;

    for (int k = 0; k < d_partition[currentLevel]; k++) {
        
        indexRoot->buckets[k] = indexBuckets[currentBucketCount];

        double leftPoint = d_minPoints[currentLevel] + k* EPSILON;
        double rightPoint = leftPoint + EPSILON;

        indexRoot->buckets[k]->range[0] = leftPoint;      
        indexRoot->buckets[k]->range[1] = rightPoint;

        currentLevel = 1;

        indexRoot->buckets[k]->level = currentLevel;
        
        for (int m = 0; m < d_partition[currentLevel]; m++) {
            currentBucketCount++;

            indexRoot->buckets[k]->buckets[m] = indexBuckets[currentBucketCount];
           
            double leftPoint = d_minPoints[currentLevel] + m* EPSILON;
            double rightPoint = leftPoint + EPSILON;

            indexRoot->buckets[k]->buckets[m]->range[0] = leftPoint;
            indexRoot->buckets[k]->buckets[m]->range[1] = rightPoint;
        }
        currentBucketCount++;
        currentLevel = 0;
    }
}

__device__
void insertData(int id, struct IndexStructure *indexRoot, struct dataNode **dataNodes, int * dataNodesLength) {

    register float data[DIMENSION];
    for (int j = 0; j < DIMENSION; j++) {
        data[j] = d_dataset[id * DIMENSION + j];
    }

    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

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
                    if(!currentIndex->buckets[k]->dataRoot) {
                        currentIndex->buckets[k]->dataRoot = dataNodes[id];
                    }
                    selectedDataNode = currentIndex->buckets[k]->dataRoot;
                    found = true;
                    break;
                }
                currentIndex = currentIndex->buckets[k];
                break;
            }
        }
    }
    
    if (selectedDataNode->id != id) {
        while (selectedDataNode->child) {
            selectedDataNode = selectedDataNode->child;
        }
        selectedDataNode->child = dataNodes[id];
        
     }
}


__device__
void searchPoints(double *data, struct IndexStructure *indexRoot) {

    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

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

    int selectedDataNodeSize = 0;

    while (currentIndexSize > 0) {

        currentIndex = currentIndexes[--currentIndexSize];

        int dimension = currentIndex->level;

        for (int k = 0; k < d_partition[dimension]; k++) {

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentIndex->buckets[k]->range[0];
            float rightRange = (float)currentIndex->buckets[k]->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {

                    selectedDataNodes[selectedDataNodeSize++] = currentIndex->buckets[k]->dataRoot;

                    if (k > 0) {
                        selectedDataNodes[selectedDataNodeSize++] = currentIndex->buckets[k - 1]->dataRoot;
                    }
                    if (k < d_partition[dimension] - 1) {
                        selectedDataNodes[selectedDataNodeSize++] = currentIndex->buckets[k + 1]->dataRoot;
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

    int resultsCount = 0;
    
    for (int x = 0; x < selectedDataNodeSize; x++) {
        selectedDataNode = selectedDataNodes[x];
        while (selectedDataNode) {
            d_results[resultsCount++] = selectedDataNode->id;
            selectedDataNode = selectedDataNode->child;
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