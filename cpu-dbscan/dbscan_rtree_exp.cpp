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

int DATASET_SIZE = 400000;

// #define DATASET_SIZE 1000

#define DIMENTION 2
#define EPSILON 0.8
#define MIN_POINTS 8

#define PORTO 0
#define SPATIAL 0
#define NGSI 0
#define IONO2D 1

using namespace std;

struct Rect {
  Rect() {}
  double min[2];
  double max[2];
  Rect(double a_minX, double a_minY, double a_maxX, double a_maxY) {
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

int importDataset(char const *fname, int N, double **dataset) {
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
  double **dataset;
  double epsilon;
  int minPoints;
  int cluster;
  int *clusters;
  double getDistance(int center, int neighbor);
  vector<int> findNeighbors(int pos);
  RTree<double, double, 2, double> tree;

 public:
  DBSCAN(double **loadData, double eps, int minPts);
  ~DBSCAN();
  void run();
  void results();
};

int main(int, char **) {
  // Generate random datasets
  char *datasetPath = "";
  double setOfR[5];
  int setOfMinPts[5];
  int defaultMin, defaultPts;
  double defaultR;
  int setOfDataSize[5];

  clock_t totalTimeStart, totalTimeStop;
  float totalTime = 0.0;
  totalTimeStart = clock();

  if (NGSI) {
    setOfDataSize[0] = 50000;
    setOfDataSize[1] = 100000;
    setOfDataSize[2] = 200000;
    setOfDataSize[3] = 400000;
    setOfDataSize[4] = 800000;

    setOfR[0] = 0.2;
    setOfR[1] = 0.4;
    setOfR[2] = 0.6;
    setOfR[3] = 0.8;
    setOfR[4] = 1;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.8;

    defaultPts = 200000;

    datasetPath = "/data/dbscan/NGSIM_Data.txt";
  }

  if (SPATIAL) {
    setOfDataSize[0] = 25000;
    setOfDataSize[1] = 50000;
    setOfDataSize[2] = 100000;
    setOfDataSize[3] = 200000;
    setOfDataSize[4] = 400000;

    setOfR[0] = 0.02;
    setOfR[1] = 0.04;
    setOfR[2] = 0.06;
    setOfR[3] = 0.08;
    setOfR[4] = 0.1;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.08;

    defaultPts = 200000;

    datasetPath = "/data/dbscan/3D_spatial_network.txt";
  }

  if (IONO2D) {
    setOfDataSize[0] = 100000;
    setOfDataSize[1] = 200000;
    setOfDataSize[2] = 400000;
    setOfDataSize[3] = 800000;
    setOfDataSize[4] = 1600000;

    setOfR[0] = 0.5;
    setOfR[1] = 0.75;
    setOfR[2] = 1;
    setOfR[3] = 1.25;
    setOfR[4] = 1.5;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 4;
    defaultR = 1.5;

    defaultPts = 200000;

    datasetPath = "/data/geodata/iono_20min_2Mpts_2D.txt";
  }

  // Different set of R
  for (int i = 0; i < 5; i++) {
    DATASET_SIZE = defaultPts;
    double **dataset = (double **)malloc(sizeof(double *) * DATASET_SIZE);
    for (int i = 0; i < DATASET_SIZE; i++) {
      dataset[i] = (double *)malloc(sizeof(double) * DIMENTION);
    }
    importDataset(datasetPath, DATASET_SIZE, dataset);
    // Initialize DBSCAN with dataset
    DBSCAN dbscan(dataset, setOfR[i], defaultMin);
    dbscan.run();

    totalTimeStop = clock();
    totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
    printf("==============================================\n");
    printf("EPS: %3.2f\nMINPTS: %d\nPOINTS: %d\n", setOfR[i], defaultMin,
           DATASET_SIZE);
    printf("Total Time: %3.2f seconds\n", totalTime);
    dbscan.results();
    printf("==============================================\n");

    for (int i = 0; i < DATASET_SIZE; i++) {
      free(dataset[i]);
    }
    free(dataset);
  }

  // Different set of MinPts
  for (int i = 0; i < 5; i++) {
    DATASET_SIZE = defaultPts;
    double **dataset = (double **)malloc(sizeof(double *) * DATASET_SIZE);
    for (int i = 0; i < DATASET_SIZE; i++) {
      dataset[i] = (double *)malloc(sizeof(double) * DIMENTION);
    }
    importDataset("/data/dbscan/NGSIM_Data.txt", DATASET_SIZE, dataset);
    // Initialize DBSCAN with dataset
    DBSCAN dbscan(dataset, defaultR, setOfMinPts[i]);
    dbscan.run();

    totalTimeStop = clock();
    totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
    printf("==============================================\n");
    printf("EPS: %3.2f\nMINPTS: %d\nPOINTS: %d\n", defaultR, setOfMinPts[i],
           DATASET_SIZE);
    printf("Total Time: %3.2f seconds\n", totalTime);
    dbscan.results();
    printf("==============================================\n");

    for (int i = 0; i < DATASET_SIZE; i++) {
      free(dataset[i]);
    }
    free(dataset);
  }

  // Different set of Points
  for (int i = 0; i < 5; i++) {
    DATASET_SIZE = setOfDataSize[i];
    double **dataset = (double **)malloc(sizeof(double *) * DATASET_SIZE);
    for (int i = 0; i < DATASET_SIZE; i++) {
      dataset[i] = (double *)malloc(sizeof(double) * DIMENTION);
    }
    importDataset("/data/dbscan/NGSIM_Data.txt", DATASET_SIZE, dataset);
    // Initialize DBSCAN with dataset
    DBSCAN dbscan(dataset, defaultR, defaultMin);
    dbscan.run();

    totalTimeStop = clock();
    totalTime = (float)(totalTimeStop - totalTimeStart) / CLOCKS_PER_SEC;
    printf("==============================================\n");
    printf("EPS: %3.2f\nMINPTS: %d\nPOINTS: %d\n", defaultR, defaultMin,
           DATASET_SIZE);
    printf("Total Time: %3.2f seconds\n", totalTime);
    dbscan.results();
    printf("==============================================\n");

    for (int i = 0; i < DATASET_SIZE; i++) {
      free(dataset[i]);
    }
    free(dataset);
  }

  return 0;
}

DBSCAN::DBSCAN(double **loadData, double eps, int minPts) {
  clusters = (int *)malloc(sizeof(int) * DATASET_SIZE);

  dataset = (double **)malloc(sizeof(double *) * DATASET_SIZE);
  for (int i = 0; i < DATASET_SIZE; i++) {
    dataset[i] = (double *)malloc(sizeof(double) * DIMENTION);
  }

  epsilon = eps;
  minPoints = minPts;
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

double DBSCAN::getDistance(int center, int neighbor) {
  double dist = (dataset[center][0] - dataset[neighbor][0]) *
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

  Rect searchRect = Rect(dataset[pos][0] - epsilon, dataset[pos][1] - epsilon,
                         dataset[pos][0] + epsilon, dataset[pos][1] + epsilon);

  searchNeighbors.clear();
  tree.Search(searchRect.min, searchRect.max, searchBoxCallback);

  for (int x = 0; x < searchNeighbors.size(); x++) {
    // Compute neighbor points of a point at position "pos"
    double distance = getDistance(pos, searchNeighbors[x]);
    if (distance <= epsilon * epsilon) {
      neighbors.push_back(searchNeighbors[x]);
    }
  }

  return neighbors;
}
