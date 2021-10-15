#include <math.h>
#include <omp.h>
#include <stdio.h>

#include <iostream>
#include <queue>

#include "read_dataset.h"
#include "report.h"

using namespace std;

#define NUM_REPORTS 3000000
report_t report[NUM_REPORTS];
int current = 0;

typedef float real;

#define BLOCK_SIZE 512

struct pointStruct {
  real x;
  real y;
};

typedef struct pointStruct point;

typedef struct {
  int device;
  int dataN;          // number of data cells assigned to this GPU thread
  point *point_Data;  // array of original point data
  int thisPoint;      // point being compared with all others
  real radius;        // distance used for comparison
  int *nOfNeighbors;  // number of neighbors
  int *clusterId;     // identifier of resulting cluster
  int nextClusterId;  // identifier for subsequent cluster
  int MinPts;         // limit under which points considered noise
  float *h_Sum;
  char *oneRow;  // results of comparision

} TGPUplan;
const int UNCLASSIFIED = -1;
const int NOISE = -2;
const int MARKED = 0;
const int MAX_NUM_GPU = 1;

int MinPts = 8;
int TOTAL_PTS = 400000;
double RADIUS = 0.08;
#define REPEAT_EXPERIMENT 1
#define PRINT_CLUSTERS 0
real Eps;  // Read from the command line
point *points;
int nOfNeighbors;

void clusterThread(TGPUplan plan) {
  // host thread constants
  const int UNCLASSIFIED = -1;
  const int NOISE = -2;
  const int MARKED = 0;

  // host thread variables
  int nOfNeighbors;
  int *clusterId;
  char *oneRow;
  int nextClusterId = 1;
  bool expandCluster(int thisPoint, int *clusterId, int nextClusterId,
                     int numPoints, real Eps, int MinPts,
                     int *numPointsInCluster, real radius, char *oneRow,
                     point *point_Data);

  clusterId = new int[plan.dataN];
  oneRow = new char[plan.dataN];

  int nonEmptyCluster = 0;

  for (int i = 0; i < plan.dataN; i++) {
    clusterId[i] = UNCLASSIFIED;
  }

  //#pragma omp parallel
  for (int i = 0; i < plan.dataN; i++) {
    if (clusterId[i] == UNCLASSIFIED) {
      int numPointsInCluster = 0;
      if (expandCluster(i, clusterId, nextClusterId, plan.dataN, RADIUS, MinPts,
                        &numPointsInCluster, RADIUS, oneRow, plan.point_Data)) {
        if (PRINT_CLUSTERS) {
          cout << "Cluster : " << nextClusterId << " contains "
               << numPointsInCluster << " points. " << endl;
        }
        if (numPointsInCluster > 0) {
          nonEmptyCluster++;
        }
        nextClusterId++;
      }
    }
  }

  set_num_clusters(&report[current], nonEmptyCluster);

  delete[] clusterId;
  delete[] oneRow;
}  // end clusterThread

int *deviceNofNeighbors;
char *resultMatrix;
void processPointCPU(int numPoints, point *masterDataArray, real radius,
                     char *resMatrix, int *nOfNeighbors, int i) {
  float myX = masterDataArray[i].x;  // set current point x
  float myY = masterDataArray[i].y;  // set current point y

  float otherX, otherY, distX, distY, Distance;
  unsigned int NumOfCandidates;

  const int numThreads = 16;
  unsigned int threadNeighbors[numThreads] = {0};
  omp_set_num_threads(numThreads);
  // cout << "Threads are " << omp_get_num_threads() << endl;

#pragma omp parallel for private(otherX, otherY, distX, distY, Distance, \
                                 NumOfCandidates)
  for (int x = 0; x < numPoints; x++) {
    NumOfCandidates = 0;

    otherX = masterDataArray[x].x;  // set all other points' x
    otherY = masterDataArray[x].y;  // set all other points' y

    distX = myX - otherX;
    distY = myY - otherY;

    // Calculate Distance
    Distance = (distX * distX) + (distY * distY);
    Distance = sqrt(Distance);

    // If this Seed is within the radius, increment number of candidates
    if (Distance <= radius) {
      NumOfCandidates++;
      resMatrix[x] = 1;
    } else {
      resMatrix[x] = 0;
    }
    // nOfNeighbors = NumOfCandidates - 1;
    if ((x != i) && (NumOfCandidates == 1)) {
      threadNeighbors[omp_get_thread_num()]++;
      //#pragma omp atomic
      //			*(nOfNeighbors) = *(nOfNeighbors)+1;
    }
  }

  for (int i = 0; i < numThreads; i++) {
    *(nOfNeighbors) = *(nOfNeighbors) + threadNeighbors[i];
  }
}

void calcRow(int thisPoint, int numPoints, real radius, char *oneRow,
             int *nOfNeighbors, report_t *report, point *point_Data) {
  int NofN = 0;
  gettimeofday(&(report->kernel_time_begin), NULL);
  *nOfNeighbors = 0;
  processPointCPU(numPoints, point_Data, radius, oneRow, nOfNeighbors,
                  thisPoint);
}

int iter = 0;
bool expandCluster(int thisPoint, int *clusterId, int nextClusterId,
                   int numPoints, real Eps, int MinPts, int *numPointsInCluster,
                   real radius, char *oneRow, point *point_Data) {
  calcRow(thisPoint, numPoints, radius, oneRow, &nOfNeighbors, &report[current],
          point_Data);

  iter++;

  if (nOfNeighbors < MinPts) {
    clusterId[thisPoint] = NOISE;
    return false;
  } else {
    (*numPointsInCluster)++;
    clusterId[thisPoint] = nextClusterId;
    queue<int> seeds;
    for (int i = 0; i < numPoints; i++) {
      if (oneRow[i] == 1) {
        seeds.push(i);
        clusterId[i] = MARKED;
      }
    }
    while (!seeds.empty()) {
      int currentPoint = seeds.front();
      // calcRow again, now for currentPoint
      calcRow(currentPoint, numPoints, radius, oneRow, &nOfNeighbors,
              &report[current], point_Data);
      seeds.pop();
      if (nOfNeighbors >= MinPts) {
        clusterId[currentPoint] = nextClusterId;
        (*numPointsInCluster)++;
        // extrapts++;
        for (int i = 0; i < numPoints; i++) {
          if (oneRow[i] == 1) {
            if (clusterId[i] == UNCLASSIFIED) {
              seeds.push(i);
            }
            clusterId[i] = MARKED;
          }
        }
      } else
        clusterId[currentPoint] = NOISE;
    }
    return true;
  }
}

void read_dataset(char *filename, traj_data_t **d_data, traj_data_t **h_data,
                  int max_pts_limit) {
  *h_data = (traj_data_t *)malloc(sizeof(traj_data_t) * 1);

  preprocess_data(filename, h_data, max_pts_limit, 1);
}

#define PORTO 1
#define SPATIAL 0
#define NGSI 0
#define IONO2D 0

////////////////////////////////////////////////////////////////
// main
////////////////////////////////////////////////////////////////

void TestRun(point *points) {
  TGPUplan plan;  // array of structs used as input for each GPU device thread
  int gpuBase;    // data start location
  plan.dataN = TOTAL_PTS;
  // Assign data ranges to GPUs:
  gpuBase = 0;
  plan.device = 1;                     // assign device GPU number
  plan.point_Data = points + gpuBase;  // set location in data array
  // plan[i].oneRow = oneRow;                 // results of the row
  plan.radius = RADIUS;               // set distance
  plan.nOfNeighbors = &nOfNeighbors;  // number of points against comparison
  // plan[i].clusterId = clusterId + gpuBase;   // identifier of cluster
  plan.MinPts = MinPts;   // minimum number of points for noise
  gpuBase += plan.dataN;  // adjust base setting

  for (int i = 0; i < REPEAT_EXPERIMENT; i++) {
    set_epsilon(&report[current], RADIUS);
    set_minPts(&report[current], MinPts);
    set_num_threads(&report[current], BLOCK_SIZE);
    set_num_blocks(&report[current], TOTAL_PTS / BLOCK_SIZE);
    set_num_points(&report[current], TOTAL_PTS);
    gettimeofday(&(report[current].processing_time_begin), NULL);
    clusterThread(plan);
    gettimeofday(&(report[current].processing_time_end), NULL);
    gettimeofday(&(report[current].total_time_end), NULL);
    print_report(&report[current]);
    current++;
  }
}
int main(int argc, char *argv[]) {
  initialize_report(&report[current]);
  init_csv();
  int max_pts_limit = 8720040;
  traj_data_t *d_data = NULL;
  traj_data_t *h_data = NULL;
  char *filename;

  if (PORTO) filename = "/data/dbscan/Porto_taxi_data.csv";
  if (SPATIAL) filename = "/data/dbscan/3D_spatial_network.txt";
  if (NGSI) filename = "/home/mpoudel/datasets/NGSIM_Data.txt";
  if (IONO2D) filename = "/data/geodata/iono_20min_2Mpts_2D.txt";

  gettimeofday(&(report[current].total_time_begin), NULL);
  gettimeofday(&(report[current].data_read_time_begin), NULL);
  read_dataset(filename, &d_data, &h_data, max_pts_limit);

  int MaxPts = 400000;
  int rangeSize = 5;
  double setOfR[5];
  int setOfMinPts[5];
  int defaultMin, defaultMinStress, defaultPts;
  double defaultR, defaultRStress;
  int setOfDataSize[5];

  if (PORTO) {
    setOfDataSize[0] = 40000;
    setOfDataSize[1] = 80000;
    setOfDataSize[2] = 160000;
    setOfDataSize[3] = 320000;
    setOfDataSize[4] = 640000;

    setOfR[0] = 0.002;
    setOfR[1] = 0.004;
    setOfR[2] = 0.006;
    setOfR[3] = 0.008;
    setOfR[4] = 0.01;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.008;

    defaultPts = 160000;
  }

  if (NGSI) {
    setOfDataSize[0] = 50000;
    setOfDataSize[1] = 100000;
    setOfDataSize[2] = 200000;
    setOfDataSize[3] = 400000;
    setOfDataSize[4] = 800000;

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

    defaultMin = 8;
    defaultR = 1.25;

    defaultPts = 400000;
  }

  if (SPATIAL) {
    setOfDataSize[0] = 25000;
    setOfDataSize[1] = 50000;
    setOfDataSize[2] = 100000;
    setOfDataSize[3] = 200000;
    setOfDataSize[4] = 400000;

    setOfR[0] = 0.002;
    setOfR[1] = 0.004;
    setOfR[2] = 0.006;
    setOfR[3] = 0.008;
    setOfR[4] = 0.01;

    setOfMinPts[0] = 4;
    setOfMinPts[1] = 8;
    setOfMinPts[2] = 16;
    setOfMinPts[3] = 32;
    setOfMinPts[4] = 64;

    defaultMin = 8;
    defaultR = 0.008;

    defaultPts = 400000;
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

    defaultPts = 400000;
  }

  points = new point[MaxPts];
  // Read from the input the x and y coordinates for each point p in the set P
  for (int i = 0; i < MaxPts; i++) {
    points[i].x = h_data->x[i];
    points[i].y = h_data->y[i];
  }
  gettimeofday(&(report[current].data_read_time_end), NULL);

  // Test impact of radius
  for (int i = 0; i < rangeSize; i++) {
    RADIUS = setOfR[i];
    MinPts = defaultMin;
    TOTAL_PTS = defaultPts;
    TestRun(points);
  }

  // Test impact of minPts
  for (int i = 0; i < rangeSize; i++) {
    RADIUS = defaultR;
    MinPts = setOfMinPts[i];
    TOTAL_PTS = defaultPts;
    TestRun(points);
  }

  // // Test impact of points
  for (int i = 0; i < rangeSize; i++) {
    RADIUS = defaultR;
    MinPts = defaultMin;
    TOTAL_PTS = setOfDataSize[i];
    TestRun(points);
  }

  cout << "cluterThread completed!" << endl;

  delete[] points;
  return 0;
}
