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

struct IndexStructure {
    int level;
    double range[2];
    struct IndexStructure *buckets[PARTITION];
    int datas[POINTS_SEARCHED];
};

int ImportDataset(char const *fname, double *dataset);

void indexConstruction(double *dataset, struct IndexStructure *indexRoot, int *partition, double minPoints[DIMENSION]);

void insertData(int id, double *data, struct IndexStructure *indexRoot, int *partition);

vector<int> searchPoints(double *data, struct IndexStructure *indexRoot, int *partition);

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

    int searchPointId = 199;
    double *data;
    data = (double *)malloc(sizeof(double) * DIMENSION);
    for (int j = 0; j < DIMENSION; j++) {
        data[j] = importedDataset[searchPointId * DIMENSION + j];
    }
    vector<int> results = searchPoints(data, indexRoot, partition);

    printf("Searched Points:\n");
    for (int i = 0; i < results.size(); i++) {
        printf("%d ", results[i]);
    }
    printf("\n");

    return 0;
}

void indexConstruction(double *dataset, struct IndexStructure *indexRoot, int *partition, double minPoints[DIMENSION]) {

    stack<struct IndexStructure *> indexStacked;
    indexStacked.push(indexRoot);

    for (int j = 0; j < DIMENSION; j++) {
        stack<struct IndexStructure *> childStacked;

        while (indexStacked.size() > 0) {
            struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentIndex = indexStacked.top();
            indexStacked.pop();
            currentIndex->level = j;

            double rightPoint = minPoints[j] + partition[j] * EPSILON;

            for (int k = partition[j] - 1; k >= 0; k--) {
                struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

                currentBucket->range[1] = rightPoint;
                rightPoint = rightPoint - EPSILON;
                currentBucket->range[0] = rightPoint;
                
                for (int i = 0; i < POINTS_SEARCHED; i++) {
                    currentBucket->datas[i] = -1;
                }
                
                currentIndex->buckets[k] = currentBucket;
                if (j < DIMENSION - 1) {
                    childStacked.push(currentIndex->buckets[k]);
                }
            }
        }

        while (childStacked.size() > 0) {
            indexStacked.push(childStacked.top());
            childStacked.pop();
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
                    for (int i = 0; i < POINTS_SEARCHED; i++) {
                        if(currentBucket->datas[i] == -1) {
                            currentBucket->datas[i] = id;
                            break;
                        }
                    }
                    found = true;
                    break;
                }
                currentIndex = currentBucket;
                break;
            }
        }
    }

}

vector<int> searchPoints(double *data, struct IndexStructure *indexRoot, int *partition) {
    struct IndexStructure *currentIndex = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));

    bool found = false;

    vector<struct IndexStructure *> currentIndexes = {};
    currentIndexes.push_back(indexRoot);

    vector<int> points = {};

    while (!currentIndexes.empty()) {

        currentIndex = currentIndexes.back();
        currentIndexes.pop_back();

        int dimension = currentIndex->level;
        for (int k = 0; k < partition[dimension]; k++) {
            struct IndexStructure *currentBucket = (struct IndexStructure *)malloc(sizeof(struct IndexStructure));
            currentBucket = currentIndex->buckets[k];

            float comparingData = (float)data[dimension];
            float leftRange = (float)currentBucket->range[0];
            float rightRange = (float)currentBucket->range[1];

            if (comparingData >= leftRange && comparingData <= rightRange) {
                if (dimension == DIMENSION - 1) {

                    for (int i = 0; i < POINTS_SEARCHED; i++) {
                        if (currentBucket->datas[i] == -1) {
                            break;
                        }
                        points.push_back(currentBucket->datas[i]);
                    }

                    if (k > 0) {

                        for (int i = 0; i < POINTS_SEARCHED; i++) {
                            if (currentIndex->buckets[k - 1]->datas[i] == -1) {
                                break;
                            }
                            points.push_back(currentIndex->buckets[k - 1]->datas[i]);
                        }
                    }
                    if (k < partition[dimension] - 1) {
                        

                        for (int i = 0; i < POINTS_SEARCHED; i++) {
                            if (currentIndex->buckets[k + 1]->datas[i] == -1) {
                                break;
                            }
                            points.push_back(currentIndex->buckets[k + 1]->datas[i]);
                        }
                    }
                    break;
                }
                currentIndexes.push_back(currentBucket);
                if (k > 0) {
                    currentIndexes.push_back(currentIndex->buckets[k - 1]);
                }
                if (k < partition[dimension] - 1) {
                    currentIndexes.push_back(currentIndex->buckets[k + 1]);
                }
                break;
            }
        }
    }

    return points;
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
        sscanf(field, "%lf", &tmp);
        dataset[cnt] = tmp;
        cnt++;

        while (field) {
            field = strtok(NULL, ",");

            if (field != NULL) {
                double tmp;
                sscanf(field, "%lf", &tmp);
                dataset[cnt] = tmp;
                cnt++;
            }
        }
    }
    fclose(fp);
    return 0;
}