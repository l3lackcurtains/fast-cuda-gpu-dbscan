#ifndef PARAMS_H
#define PARAMS_H

using namespace std;

// Number of data in dataset to use
#define DATASET_COUNT 1864620
// #define DATASET_COUNT 50000000

// Dimension of the dataset
#define DIMENSION 2

// Maximum size of seed list
#define MAX_SEEDS 256

// Extra collission size to detect final clusters collision
#define EXTRA_COLLISION_SIZE 512

// Number of blocks
#define THREAD_BLOCKS 128

// Number of threads per block
#define THREAD_COUNT 256

// Status of points that are not clusterized
#define UNPROCESSED -1

// Status for noise point
#define NOISE -2

// Minimum number of points in DBSCAN
#define MINPTS 4

#define TREE_LEVELS (DIMENSION + 1)

// Epslion value in DBSCAN
#define EPS 1.5

#define RANGE 2

#define POINTS_SEARCHED 9

#define PARTITION_SIZE 100

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

struct __align__(8) IndexStructure {
  int dimension;
  int dataBegin;
  int dataEnd;
  int childFrom;
};


#endif