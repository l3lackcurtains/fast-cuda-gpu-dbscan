#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#define DATASET_SIZE 10000

using namespace std;

void DetermineErrorTwoClusterResults(vector<int> *c1, vector<int> *c2);

int importDataset(char const *fname, int N, int *cluster);

int main() {
  int *clusterData1 = (int *)malloc(sizeof(int) * DATASET_SIZE);
  int *clusterData2 = (int *)malloc(sizeof(int) * DATASET_SIZE);

  importDataset("../out/cpu_dbscan_output.txt", DATASET_SIZE, clusterData1);
  importDataset("../out/gpu_dbscan_output.txt", DATASET_SIZE, clusterData2);

  vector<int> cls1;
  vector<int> cls2;

  for (int i = 0; i < DATASET_SIZE; i++) {
    cls1.push_back(clusterData1[i]);
    cls2.push_back(clusterData2[i]);
  }

  DetermineErrorTwoClusterResults(&cls1, &cls2);

  return 0;
}

void DetermineErrorTwoClusterResults(std::vector<int> *c1,
                                     std::vector<int> *c2) {
  printf("\nIn method to compare two clusters similarity metric");
  cout.flush();

  printf(
      "\nsize of array of datapoints1: %zu, size of array of datapoints2: %zu",
      c1->size(), c2->size());
  cout.flush();

  if (c1->size() != c2->size()) {
    printf(
        "\n**********\nERROR WHEN TESTING THE SIMILARITY/ERROR OF TWO "
        "CLUSTERING RESULTS. THE NUMBER OF POINTS IN EACH ARRAY ARE NOT "
        "EQUAL\n\n");
    return;
  }

  const int sizeData = int(c1->size());
  printf("\nsize of data in var: %d", sizeData);

  double *scoreArr;
  scoreArr = new double[sizeData];

  bool *visitedArr;
  visitedArr = new bool[sizeData];

  // initialize:
  for (int i = 0; i < sizeData; i++) {
    scoreArr[i] = 0;
    visitedArr[i] = false;
  }

  int max1 = 0;

  for (int i = 0; i < sizeData; i++) {
    if (((*c1)[i]) > max1) {
      max1 = (*c1)[i];
    }
  }

  int max2 = 0;
  for (int i = 0; i < c2->size(); i++) {
    // the max cluster id in the first one
    if (((*c2)[i]) > max2) {
      max2 = (*c2)[i];
    }
  }

  max1++;
  max2++;

  printf("\nnum clusters in list1, list2: %d,%d", max1, max2);
  cout.flush();

  std::vector<int> clusterArr1[max1];
  std::vector<int> clusterArr2[max2];

  for (int i = 0; i < sizeData; i++) {
    int clusterid1 = (*c1)[i];
    int clusterid2 = (*c2)[i];

    clusterArr1[clusterid1].push_back(i);
    clusterArr2[clusterid2].push_back(i);
  }

  // sort the array of clusters for cluster set 1
  for (int i = 0; i < max1; i++) {
    std::sort(clusterArr1[i].begin(), clusterArr1[i].end());
  }

  // sort the array of clusters for cluster set 2
  for (int i = 0; i < max2; i++) {
    std::sort(clusterArr2[i].begin(), clusterArr2[i].end());
  }

  int cntNoiseError = 0;
  int cntNoiseEqual = 0;

  for (int i = 0; i < sizeData; i++) {
    if (((*c1)[i] == 0) && ((*c2)[i] != 0)) {
      cntNoiseError++;
      scoreArr[i] = 0;
      visitedArr[i] = true;
    }

    if (((*c1)[i] != 0) && ((*c2)[i] == 0)) {
      cntNoiseError++;
      scoreArr[i] = 0;
      visitedArr[i] = true;
    }

    // both are noise points:
    if (((*c1)[i] == 0) && ((*c2)[i] == 0)) {
      cntNoiseEqual++;
      scoreArr[i] = 1;
      visitedArr[i] = true;
    }
  }

  printf("\nmismatched noise points: %d, agreement noise points: %d",
         cntNoiseError, cntNoiseEqual);

  double totalScoreMiscluster = 0;
  printf("\nRUNNING THE VALIDATION IN PARALLEL WITH THREADS!!");

  for (int i = 0; i < sizeData; i++) {
    // if point was already noise, we dealt with the point already
    if (visitedArr[i] == true) {
      continue;
    }

    // get the two cluster ids for the point from the two experiments
    int clusterid1 = (*c1)[i];
    int clusterid2 = (*c2)[i];

    std::vector<int> ids_in_cluster1 = clusterArr1[clusterid1];
    std::vector<int> ids_in_cluster2 = clusterArr2[clusterid2];

    int cntIntersection = 0;
    int cntUnion = 0;

    for (int j = 0; j < ids_in_cluster1.size(); j++) {
      int findElem = ids_in_cluster1[j];
      if (std::binary_search(ids_in_cluster2.begin(), ids_in_cluster2.end(),
                             findElem)) {
        cntIntersection++;
      }
    }

    int preallocateUnion = ids_in_cluster1.size() + ids_in_cluster2.size();
    std::vector<int> unionSet(preallocateUnion);
    std::vector<int>::iterator it2;

    // get the union of the two clusters and store in the unionSet vector
    it2 = std::set_union(ids_in_cluster1.begin(), ids_in_cluster1.end(),
                         ids_in_cluster2.begin(), ids_in_cluster2.end(),
                         unionSet.begin());
    unionSet.resize(it2 - unionSet.begin());

    scoreArr[i] = (cntIntersection * 1.0) / (unionSet.size() * 1.0);

    // for testing:
    if (scoreArr[i] != 1.0) {
      totalScoreMiscluster += 1.0 - scoreArr[i];
    }
  }

  printf("\nfraction lost due to mismatches between clusters: %f",
         totalScoreMiscluster / (sizeData * 1.0));

  // final score:
  double sum = 0;
  for (int i = 0; i < sizeData; i++) {
    sum += scoreArr[i];
  }

  printf("\nFinal Error metric: %f", (1.0 * sum) / (sizeData * 1.0));
}

int importDataset(char const *fname, int N, int *cluster) {
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
    int tmp;
    sscanf(field, "%d", &tmp);
    cluster[rowCnt] = tmp;
    rowCnt++;
  }

  fclose(fp);

  return 0;
}