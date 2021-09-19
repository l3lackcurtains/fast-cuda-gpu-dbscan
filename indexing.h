#ifndef INDEXING_H
#define INDEXING_H

__global__ void INDEXING_ADJUSTMENT(int *indexTreeMetaData,
                                    struct IndexStructure **indexBuckets,
                                    int *dataKey);

__global__ void INDEXING_STRUCTURE(double *dataset, int *indexTreeMetaData,
                                        double *minPoints, double *maxPoints,
                                        double *binWidth, int *results,
                                        struct IndexStructure **indexBuckets,
                                        int *dataKey, int *dataValue,
                                        double *upperBounds);

__device__ void insertData(int id, double *dataset,
                                struct IndexStructure **indexBuckets,
                                int *dataKey, int *dataValue,
                                double *upperBounds, double *binWidth,
                                double *minPoints, double *maxPoints);

__device__ void indexConstruction(int level, int *indexTreeMetaData,
                                       double *minPoints, double *binWidth,
                                       struct IndexStructure **indexBuckets,
                                       double *upperBounds);

__device__ void searchPoints(double *data, int chainID, double *dataset,
                                  int *results,
                                  struct IndexStructure **indexBuckets,
                                  int *indexesStack, int *dataValue,
                                  double *upperBounds, double *binWidth,
                                  double *minPoints, double *maxPoints);
#endif