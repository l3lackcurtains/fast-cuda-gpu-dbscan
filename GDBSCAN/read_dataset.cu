#include <stdio.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

#include "read_dataset.h"

#define DIMENSION 2

std::vector<float> lon_ptr;
std::vector<float> lat_ptr;
std::vector<int> PTI;
std::vector<int> TKI;

//#define CPU
using namespace std;

int read_data_traj_per_line(char* filename, int max_pts_limit,
                            traj_data_t** data, int repeat) {
  FILE* fp_data = fopen(filename, "rb");
  char line[10000];

  int lineNo = 0;
  int wordNo = 0;

  int trajID = 0;
  int ptsInTraj = 0;
  int ptsInTraj2 = 0;

  PTI.push_back(0);
  TKI.push_back(0);

  char buf[4096];
  unsigned long int cnt = 0;

  while (fgets(buf, 4096, fp_data) && cnt < max_pts_limit * DIMENSION) {
    char* field = strtok(buf, ",");
    long double tmp;
    sscanf(field, "%Lf", &tmp);
    lon_ptr.push_back(tmp);
    ptsInTraj++;
    cnt++;

    while (field) {
      field = strtok(NULL, ",");

      if (field != NULL) {
        long double tmp;

        sscanf(field, "%Lf", &tmp);

        lat_ptr.push_back(tmp);
        ptsInTraj2++;
        cnt++;
      }
    }
  }
  fclose(fp_data);

  int num_points = lon_ptr.size();

  printf("Number of trajectories are %d\n", PTI.size());
  printf("Number of points is %d\n", lon_ptr.size());
  *data = (traj_data_t*)malloc(sizeof(traj_data_t) * 1);
  (*data)->TKI = NULL;
  (*data)->TKI_length = 0;
  (*data)->PTI = NULL;
  (*data)->PTI_length = 0;
  (*data)->x = static_cast<float*>(&lon_ptr[0]);
  (*data)->y = static_cast<float*>(&lat_ptr[0]);
  (*data)->num_points = num_points * repeat;
  (*data)->track_to_traj = NULL;
  (*data)->num_tracks = NULL;

  (*data)->grid.top_left_corner_x =
      *(min_element(lon_ptr.begin(), lon_ptr.end()));
  (*data)->grid.top_left_corner_y =
      *(max_element(lat_ptr.begin(), lat_ptr.end()));

  (*data)->grid.bottom_right_corner_x =
      *(max_element(lon_ptr.begin(), lon_ptr.end()));
  (*data)->grid.bottom_right_corner_y =
      *(min_element(lat_ptr.begin(), lat_ptr.end()));

  printf("[INFO] Grid top left = (%f,%f)\n", (*data)->grid.top_left_corner_x,
         (*data)->grid.top_left_corner_y);
  printf("[INFO] Grid bottom right = (%f,%f)\n",
         (*data)->grid.bottom_right_corner_x,
         (*data)->grid.bottom_right_corner_y);

  return num_points;
}

int read_data_traj_per_line_3d(char* filename, int max_pts_limit, traj_data_t** data, int repeat)
{
 
    ifstream file ( filename );
 
    int lineNo = 0;
    int wordNo = 0;
    int lonlatno = 100;
    bool badLongitude = false;
 
    int trajID=0;
    int ptsInTraj=0;
 
    PTI.push_back(0);
    TKI.push_back(0);
 
    string thisWord;
	while (file.good()) {
		getline ( file, thisWord, ',' );
		getline ( file, thisWord, ',' );
		lon_ptr.push_back(stof(thisWord));
		getline ( file, thisWord, ',' );
		lat_ptr.push_back(stof(thisWord));
	}
	lon_ptr.pop_back();
	lat_ptr.pop_back();
 
	int num_points = lon_ptr.size();
	PTI.pop_back();

	printf("Number of trajectories are %d\n",PTI.size());
	printf("Number of points is %d\n",lon_ptr.size());
	*data = (traj_data_t*) malloc(sizeof(traj_data_t) * 1);
	(*data)->TKI           = NULL;
	(*data)->TKI_length    = 0;
	(*data)->PTI           = static_cast<int*>(& PTI[0]);
	(*data)->PTI_length    = PTI.size();
	(*data)->x             = static_cast<float*>(& lon_ptr[0]);
	(*data)->y             = static_cast<float*>(& lat_ptr[0]);
	(*data)->num_points    = num_points*repeat;
	(*data)->track_to_traj = NULL;
	(*data)->num_tracks    = NULL;
 
 
    (*data)->grid.top_left_corner_x = *(min_element(lon_ptr.begin(), lon_ptr.end()));
    (*data)->grid.top_left_corner_y = *(max_element(lat_ptr.begin(), lat_ptr.end()));
 
    (*data)->grid.bottom_right_corner_x = *(max_element(lon_ptr.begin(), lon_ptr.end()));
    (*data)->grid.bottom_right_corner_y = *(min_element(lat_ptr.begin(), lat_ptr.end()));
 
    printf("[INFO] Grid top left = (%f,%f)\n",  (*data)->grid.top_left_corner_x,
           (*data)->grid.top_left_corner_y );
    printf("[INFO] Grid bottom right = (%f,%f)\n",
           (*data)->grid.bottom_right_corner_x,
           (*data)->grid.bottom_right_corner_y);
 
    return 0;
}

int preprocess_data(char* filename, traj_data_t** data, int max_pts_limit,
                    int repeat) {

  vector<Point> h_points;
  int ret = -1;
  char* rest = (char*)malloc(sizeof(char) * 254);
  ;
  strcpy(rest, filename);
  char *token, *dataSetName;
  while ((token = strtok_r(rest, "/", &rest))) {
    dataSetName = token;
  }


  printf("[INFO] Using 2D dataset.\n");
  int num_points =
      read_data_traj_per_line(filename, max_pts_limit, data, repeat);
  ret = 0;

  return ret;
}
