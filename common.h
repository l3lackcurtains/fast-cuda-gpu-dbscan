#ifndef PARAMS_H
#define PARAMS_H

using namespace std;

#define RANGE 2
#define UNPROCESSED -1
#define NOISE -2

#define DIMENSION 2
#define TREE_LEVELS (DIMENSION + 1)

#define THREAD_BLOCKS 8
#define THREAD_COUNT 16

#define MAX_SEEDS 128
#define EXTRA_COLLISION_SIZE 256

#define DATASET_COUNT 9

#define MINPTS 3
#define EPS 1.5

#define PARTITION_SIZE 3
#define POINTS_SEARCHED 9

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
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