#ifndef _MAKE_GRAPH_H_
#define _MAKE_GRAPH_H_

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "breadthFirstSearch.h"
#include "report.h"

using namespace std;

struct Graph {
  long unsigned int *nodes;
  int *edges;
  int totalEdges;
};

void makeGraph(int NUM_BLOCKS, int BLOCK_THREADS, float *x, float *y,
               int numPoints, int minPts, float R, Graph *distGraph,
               bool **clusterType, report_t *report);

__global__ void fillNodes(int minPts, float R, int numPoints, float *d_x,
                          float *d_y, long unsigned int *dNodes,
                          bool *dClusterType);

__global__ void fillEdges(int numPoints, float R, float *d_x, float *d_y,
                          long unsigned int *dNodes, int *dEdges);

__device__ __host__ float euclidean_distance(float p1_x, float p1_y, float p2_x,
                                             float p2_y);

#endif
