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

void indexConstruction(double *dataset, struct IndexStructure *indexRoot, int *partition, double minPoints[DIMENSION]);

void insertData(int id, double *data, struct IndexStructure *indexRoot, int *partition);

void searchPoints(int * results, double *data, struct IndexStructure *indexRoot, int *partition);

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

    indexConstruction(importedDataset, indexRoot, partition, minPoints);

    int searchPointId = 100;
    double *data;
    data = (double *)malloc(sizeof(double) * DIMENSION);
    for (int j = 0; j < DIMENSION; j++) {
        data[j] = importedDataset[searchPointId * DIMENSION + j];
    }

    int * results = (int *) malloc(sizeof(int) * POINTS_SEARCHED);
    for(int i = 0; i < POINTS_SEARCHED; i++) {
      results[i] = -1;
    }
    searchPoints(results, data, indexRoot, partition);

    printf("Searched Points:\n");
    for (int i = 0; i < POINTS_SEARCHED; i++) {
        if(results[i] == -1) break;
        printf("%d ", results[i]);
    }
    printf("\n");

    return 0;
}

void indexConstruction(double *dataset, struct IndexStructure *indexRoot, int *partition, double minPoints[DIMENSION]) {

    int indexedStructureSize = 0;
    for(int i = 0; i < DIMENSION; i++) {
      indexedStructureSize += partition[i];
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

            double rightPoint = minPoints[j] + partition[j] * EPSILON;

            for (int k = partition[j] - 1; k >= 0; k--) {
                struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

                currentBucket->range[1] = rightPoint;
                rightPoint = rightPoint - EPSILON;
                currentBucket->range[0] = rightPoint;
                currentBucket->dataRoot = (struct dataNode *)malloc(sizeof(struct dataNode));
                currentBucket->dataRoot->id = -1;

                currentIndex->buckets[k] = currentBucket;
                if (j < DIMENSION - 1) {
                    childIndexedStructures[childIndexedStructureSizeCount++] = currentIndex->buckets[k];
                }
            }
        }

        while (childIndexedStructureSizeCount > 0) {
          indexedStructures[indexedStructureSizeCount++] = childIndexedStructures[--childIndexedStructureSizeCount];
        }
    }

    for (int i = 0; i < DATASET_COUNT; i++) {
        double *data =
            (double *)malloc(sizeof(double) * DIMENSION);
        for (int j = 0; j < DIMENSION; j++) {
            data[j] = dataset[i * DIMENSION + j];
        }
        insertData(i, data, indexRoot, partition);
    }
}

void insertData(int id, double *data, struct IndexStructure *indexRoot, int *partition) {
    
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

    currentIndex = indexRoot;
    bool found = false;

    while (!found) {
        int dimension = currentIndex->level;
        for (int k = 0; k < partition[dimension]; k++) {
            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentBucket = currentIndex->buckets[k];

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentBucket->range[0];
            float rightRange = (float)currentBucket->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {
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

void searchPoints(int * results, double *data, struct IndexStructure *indexRoot, int *partition) {
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    struct dataNode *selectedDataNode = (struct dataNode *)malloc(sizeof(struct dataNode));

    // Size of data Node and index
    int indexedStructureSize = 1;
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

        for (int k = 0; k < partition[dimension]; k++) {

            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

            currentBucket = currentIndex->buckets[k];

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentBucket->range[0];
            float rightRange = (float)currentBucket->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {
                    selectedDataNodes[selectedDataNodeSize++] = currentBucket->dataRoot;
                    if (k > 0) {
                        selectedDataNodes[selectedDataNodeSize++] = currentIndex->buckets[k - 1]->dataRoot;
                    }
                    if (k < partition[dimension] - 1) {
                        selectedDataNodes[selectedDataNodeSize++] = currentIndex->buckets[k + 1]->dataRoot;
                    }
                    break;
                }
                currentIndexes[currentIndexSize++] = currentBucket;
                if (k > 0) {
                    currentIndexes[currentIndexSize++] = currentIndex->buckets[k - 1];
                }
                if (k < partition[dimension] - 1) {
                    currentIndexes[currentIndexSize++] = currentIndex->buckets[k + 1];
                }
                break;
            }
        }
    }

    int resultsCount = 0;
    for (int x = 0; x < selectedDataNodeSize; x++) {
        selectedDataNode = selectedDataNodes[x];
        while (selectedDataNode->id != -1) {
            results[resultsCount++] = selectedDataNode->id;
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