#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vector>

#define DATASET_COUNT 1000
#define DIMENSION 2
#define PARTITION 10

#define EPSILON 1.5

using namespace std;

struct dataNode {
    int id;
    struct dataNode *child;
};

struct IndexStructure {
    int level;
    double range[2];
    struct IndexStructure *buckets[PARTITION];
    struct dataNode *dataRoot;
};

int ImportDataset(char const *fname, double *dataset);

void indexConstruction(double *dataset, struct IndexStructure *indexRoot);

void insertData(int id, double data[DIMENSION], struct IndexStructure *indexRoot);

vector<int> searchPoints(double data[DIMENSION], struct IndexStructure *indexRoot);

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

    struct IndexStructure *indexRoot = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    indexConstruction(importedDataset, indexRoot);

    int searchPointId = 12;
    double data[DIMENSION];
    for (int j = 0; j < DIMENSION; j++) {
        data[j] = importedDataset[searchPointId * DIMENSION + j];
    }
    vector<int> results = searchPoints(data, indexRoot);

    printf("Searched Points:\n");
    for (int i = 0; i < results.size(); i++) {
        printf("%d ", results[i]);
    }
    printf("\n");

    return 0;
}

void indexConstruction(double *dataset, struct IndexStructure *indexRoot) {
    double maxPoints[DIMENSION];
    double minPoints[DIMENSION];

    for (int j = 0; j < DIMENSION; j++) {
        maxPoints[j] = 0;
        minPoints[j] = 999999999;
    }

    for (int i = 0; i < DATASET_COUNT; i++) {
        for (int j = 0; j < DIMENSION; j++) {
            if (dataset[i * DIMENSION + j] > maxPoints[j]) {
                maxPoints[j] = dataset[i * DIMENSION + j];
            }
            if (dataset[i * DIMENSION + j] < minPoints[j]) {
                minPoints[j] = dataset[i * DIMENSION + j];
            }
        }
    }

    double advanceRangeBy[DIMENSION];

    for (int j = 0; j < DIMENSION; j++) {
        advanceRangeBy[j] = (maxPoints[j] - minPoints[j]) / PARTITION;
    }

    stack<struct IndexStructure *> indexStacked;
    indexStacked.push(indexRoot);

    for (int j = 0; j < DIMENSION; j++) {
        stack<struct IndexStructure *> childStacked;

        while (indexStacked.size() > 0) {
            struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentIndex = indexStacked.top();
            indexStacked.pop();

            double rightPoint = maxPoints[j];
            for (int k = PARTITION - 1; k >= 0; k--) {
                struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

                currentBucket->range[1] = rightPoint;
                rightPoint = rightPoint - advanceRangeBy[j];
                currentBucket->range[0] = rightPoint;
                currentBucket->level = j;
                currentBucket->dataRoot = (struct dataNode *)malloc(sizeof(struct dataNode));
                currentBucket->dataRoot->id = -1;

                currentIndex->buckets[k] = currentBucket;
                childStacked.push(currentIndex->buckets[k]);
            }
        }
        while (childStacked.size() > 0) {
            indexStacked.push(childStacked.top());
            childStacked.pop();
        }
    }

    for (int i = 0; i < DATASET_COUNT; i++) {
        double data[DIMENSION];
        for (int j = 0; j < DIMENSION; j++) {
            data[j] = dataset[i * DIMENSION + j];
        }
        insertData(i, data, indexRoot);
    }
}

vector<int> searchPoints(double data[DIMENSION], struct IndexStructure *indexRoot) {
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

    currentIndex = indexRoot;
    bool found = false;
    while (!found) {
        for (int k = 0; k < PARTITION; k++) {
            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentBucket = currentIndex->buckets[k];

            float comparingData = (float) data[currentBucket->level];
            float leftRange = (float) currentBucket->range[0];
            float rightRange = (float) currentBucket->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (currentBucket->level == DIMENSION - 1) {
                    selectedDataNode = currentBucket->dataRoot;
                    found = true;
                    break;
                }
                currentIndex = currentBucket;
                break;
            }
        }
    }

    vector<int> points = {};
    while (selectedDataNode->id != -1) {
        
        points.push_back(selectedDataNode->id);
        selectedDataNode = selectedDataNode->child;
    }
    return points;
}

void insertData(int id, double data[DIMENSION], struct IndexStructure *indexRoot) {
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

    currentIndex = indexRoot;
    bool found = false;

    while (!found) {
        for (int k = 0; k < PARTITION; k++) {
            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentBucket = currentIndex->buckets[k];

           float comparingData = (float)data[currentBucket->level];
            float leftRange = (float)currentBucket->range[0];
            float rightRange = (float)currentBucket->range[1];
            
            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (currentBucket->level == DIMENSION - 1) {
                    selectedDataNode = currentBucket->dataRoot;
                    found = true;
                    break;
                }
                currentIndex = currentBucket;
                break;
            }
        }
    }
    if (selectedDataNode->id == -1) {
        selectedDataNode->id = id;
        selectedDataNode->child = (struct dataNode *)malloc(sizeof(struct dataNode));
        selectedDataNode->child->id = -1;
    } else {
        selectedDataNode = selectedDataNode->child;
        while (selectedDataNode->id != -1) {
            selectedDataNode = selectedDataNode->child;
        }
        selectedDataNode->id = id;
        selectedDataNode->child = (struct dataNode *)malloc(sizeof(struct dataNode));
        selectedDataNode->child->id = -1;
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
        double tmp;
        sscanf(field, "%Lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;

        while (field) {
            field = strtok(NULL, ",");

            if (field != NULL) {
                double tmp;
                sscanf(field, "%Lf", &tmp);
                dataset[cnt] = tmp;
                cnt++;
            }
        }
    }
    fclose(fp);
    return 0;
}