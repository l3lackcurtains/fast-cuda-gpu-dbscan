#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Rtree.h"

#define DATASET_SIZE 1864620

// #define DATASET_SIZE 1000

#define DIMENTION 2
#define ELIPSON 1.25
#define MIN_POINTS 4

using namespace std;

struct Rect {
  Rect() {}
  long double min[2];
  long double max[2];
  Rect(long double a_minX, long double a_minY, long double a_maxX,
       long double a_maxY) {
    min[0] = a_minX;
    min[1] = a_minY;

    max[0] = a_maxX;
    max[1] = a_maxY;
  }
};

vector<int> searchNeighbors;
bool searchBoxCallback(int id) {
  searchNeighbors.push_back(id);
  return true;
}

int importDataset(char const *fname, int N, long double **dataset) {
  FILE *fp = fopen(fname, "r");

  if (!fp) {
    printf("Unable to open file\n");
    return (1);
  }

  char buf[4096];
  int rowCnt = 0;
  int colCnt = 0;
  while (fgets(buf, 4096, fp) && rowCnt < N) {
    colCnt = 0;

    char *field = strtok(buf, ",");
    long double tmp;
    sscanf(field, "%Lf", &tmp);
    dataset[rowCnt][colCnt] = tmp;

    while (field) {
      colCnt++;
      field = strtok(NULL, ",");

      if (field != NULL) {
        long double tmp;
        sscanf(field, "%Lf", &tmp);
        dataset[rowCnt][colCnt] = tmp;
      }
    }
    rowCnt++;
  }

  fclose(fp);

  return 0;
}

class DBSCAN {
 private:
  long double **dataset;
  double elipson;
  int minPoints;
  int cluster;
  int *clusters;
  long double getDistance(int center, int neighbor);
  vector<int> findNeighbors(int pos);
  RTree<long double, long double, 2, long double> tree;

 public:
  DBSCAN(long double **loadData);
  ~DBSCAN();
  void run();
  void results();
};

int main(int, char **) {
  // Generate random datasets
  long double **dataset =
      (long double **)malloc(sizeof(long double *) * DATASET_SIZE);
  for (int i = 0; i < DATASET_SIZE; i++) {
    dataset[i] = (long double *)malloc(sizeof(long double) * DIMENTION);
  }

  importDataset("../dataset/dataset.txt", DATASET_SIZE, dataset);


  clock_t totalTimeStart, totalTimeStop;
  float totalTime = 0.0;
  totalTimeStart = clock();

  // Initialize DBSCAN with dataset
  DBSCAN dbscan(dataset);

  // Run the DBSCAN algorithm
  dbscan.run();

  totalTimeStop = clock();
  totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
  printf("==============================================\n");
  printf("Total Time: %3.2f seconds\n", totalTime);
  printf("==============================================\n");

    // Print the cluster results of DBSCAN
  dbscan.results();
  
  for (int i = 0; i < DATASET_SIZE; i++) {
    free(dataset[i]);
  }
  free(dataset);

  return 0;
}

DBSCAN::DBSCAN(long double **loadData) {
  clusters = (int *)malloc(sizeof(int) * DATASET_SIZE);

  dataset = (long double **)malloc(sizeof(long double *) * DATASET_SIZE);
  for (int i = 0; i < DATASET_SIZE; i++) {
    dataset[i] = (long double *)malloc(sizeof(long double) * DIMENTION);
  }

  elipson = ELIPSON;
  minPoints = MIN_POINTS;
  cluster = 0;

  for (int i = 0; i < DATASET_SIZE; i++) {
    dataset[i][0] = loadData[i][0];
    dataset[i][1] = loadData[i][1];
    clusters[i] = 0;

    // Insert Data into tree
    Rect rectange =
        Rect(dataset[i][0], dataset[i][1], dataset[i][0], dataset[i][1]);
    tree.Insert(rectange.min, rectange.max, i);
  }
}

DBSCAN::~DBSCAN() {
  for (int i = 0; i < DATASET_SIZE; i++) {
    free(dataset[i]);
  }
  free(clusters);
  free(dataset);
}

long double DBSCAN::getDistance(int center, int neighbor) {
  long double dist = (dataset[center][0] - dataset[neighbor][0]) *
                         (dataset[center][0] - dataset[neighbor][0]) +
                     (dataset[center][1] - dataset[neighbor][1]) *
                         (dataset[center][1] - dataset[neighbor][1]);

  return dist;
}

void DBSCAN::run() {
  // Neighbors of the point

  for (int i = 0; i < DATASET_SIZE; i++) {
    if (clusters[i] != 0) continue;
    vector<int> neighbors;
    // Find neighbors of point P
    neighbors = findNeighbors(i);

    // Mark noise points
    if (neighbors.size() < minPoints) {
      clusters[i] = -1;
      continue;
    }
    cluster++;
    // Increment cluster and initialize it will the current point
    clusters[i] = cluster;

    // Expand the neighbors of point P
    for (int j = 0; j < neighbors.size(); j++) {
      // Mark neighbour as point Q
      int dataIndex = neighbors[j];

      if (dataIndex == i) continue;

      if (clusters[dataIndex] == -1) {
        clusters[dataIndex] = cluster;
        continue;
      }
      if (clusters[dataIndex] != 0) continue;

      clusters[dataIndex] = cluster;

      // Expand more neighbors of point Q
      vector<int> moreNeighbors;
      moreNeighbors = findNeighbors(dataIndex);

      // Continue when neighbors point is higher than minPoint threshold
      if (moreNeighbors.size() >= minPoints) {
        // Check if neighbour of Q already exists in neighbour of P
        for (int x = 0; x < moreNeighbors.size(); x++) {
          neighbors.push_back(moreNeighbors[x]);
        }
      }
    }
  }
}

void DBSCAN::results() {
  printf("Number of clusters: %d\n", cluster);
  int noises = 0;
  for (int i = 0; i < DATASET_SIZE; i++) {
    if (clusters[i] == -1) {
      noises++;
    }
  }

  printf("Noises: %d\n", noises);
}

vector<int> DBSCAN::findNeighbors(int pos) {
  vector<int> neighbors;

  Rect searchRect = Rect(dataset[pos][0] - elipson, dataset[pos][1] - elipson,
                         dataset[pos][0] + elipson, dataset[pos][1] + elipson);

  searchNeighbors.clear();
  tree.Search(searchRect.min, searchRect.max, searchBoxCallback);

  for (int x = 0; x < searchNeighbors.size(); x++) {
    // Compute neighbor points of a point at position "pos"
    long double distance = getDistance(pos, searchNeighbors[x]);
    if (distance <= elipson * elipson) {
      neighbors.push_back(searchNeighbors[x]);
    }
  }

  return neighbors;
}
