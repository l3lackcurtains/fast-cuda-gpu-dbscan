#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#define DATASET_SIZE 1000
using namespace std;

int importResults(char const *fname, int N, int *cluster);

int main() {
  int *clusterData1 = (int *)malloc(sizeof(int) * DATASET_SIZE);
  int *clusterData2 = (int *)malloc(sizeof(int) * DATASET_SIZE);

  importResults("./out/cuda_dclust_extended.txt", DATASET_SIZE, clusterData1);
  importResults("./finalized-dbscan-indexing/out/cuda_dclust.txt", DATASET_SIZE,
                clusterData2);

  int max1 = *max_element(clusterData1, clusterData1 + DATASET_SIZE);
  int max2 = *max_element(clusterData2, clusterData2 + DATASET_SIZE);

  vector<vector<int>> cls1;
  vector<vector<int>> cls2;

  for (int i = 1; i <= max1; i++) {
    vector<int> clusterr;
    for (int j = 0; j < DATASET_SIZE; j++) {
      if (i == clusterData1[j]) {
        clusterr.push_back(j);
      }
    }
    cls1.push_back(clusterr);
  }

  for (int i = 1; i <= max2; i++) {
    vector<int> clusterr;
    for (int j = 0; j < DATASET_SIZE; j++) {
      if (i == clusterData2[j]) {
        clusterr.push_back(j);
      }
    }
    cls2.push_back(clusterr);
  }

  ofstream outputFile;
  outputFile.open("cpu_cls_out.txt");
  for (int i = 0; i < max1; i++) {
    outputFile << i + 1 << ":" << endl;
    for (int j = 0; j < cls1[i].size(); j++) {
      outputFile << cls1[i][j] << ",";
    }
    outputFile << endl << endl;
  }

  outputFile.close();

  outputFile.open("gpu_cls_out.txt");
  for (int i = 0; i < max2; i++) {
    outputFile << i + 1 << ":" << endl;
    for (int j = 0; j < cls2[i].size(); j++) {
      outputFile << cls2[i][j] << ",";
    }
    outputFile << endl << endl;
  }

  outputFile.close();

  return 0;
}

int importResults(char const *fname, int N, int *cluster) {
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