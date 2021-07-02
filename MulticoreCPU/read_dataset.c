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

#define DIMENSION 2

#include "read_dataset.h"

#define XMIN 11610000
#define YMIN 3970000
#define XMAX 11675530
#define YMAX 4035530
#define SCALE 1

std::vector<float> lon_ptr;
std::vector<float> lat_ptr;
std::vector<int>  PTI;
std::vector<int>  TKI;

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


int read_data_traj_per_line_porto(char* filename, int max_pts_limit, traj_data_t** data, int repeat)
{

	//This function reads and segments trajectories in dataset in the following format:
	//The first line indicates number of variables per point (I'm ignoring that and assuming 2)
	//The second line indicates total trajectories in file (I'm ignoring that and
	//observing how many are there by reading them). All lines that follow contains a
	//trajectory separated by new line.
	//The first number in the trajectory is the number of points followed by
	//location points separated by spaces

    FILE *fp_data = fopen(filename, "rb");
    char line[100000];

    int lineNo = -1;
    int wordNo = 0;
    int lonlatno = 100;
    bool badLongitude = false;

    int trajID=0;
    int ptsInTraj=0;

    PTI.push_back(0);
    TKI.push_back(0);

    float thisWord;
	while (fgets(line, sizeof(line), fp_data)) {
		if(lineNo>-1){
			char * pch;
			char *end_str;
			wordNo = 0;
			lonlatno=0;
			badLongitude = false;
			pch = strtok_r(line, "\"[", &end_str);
			while (pch != NULL)
			{
				if(wordNo>0){
					char * pch2;
					char *end_str2;

					pch2 = strtok_r(pch,",",&end_str2);

					if(strcmp(pch2,"]")<0 && lonlatno<255){

						thisWord = atof(pch2);

						if(thisWord != 0.00000){
							if(thisWord>-9 && thisWord<-7){
								lon_ptr.push_back(thisWord);
							//printf("lon %f",thisWord);

								pch2 = strtok_r(NULL,",",&end_str2);
								thisWord = atof(pch2);
								if(thisWord < 42 && thisWord > 40){
									lat_ptr.push_back(thisWord);
									//printf(" lat %f\n",thisWord);

									lonlatno++;
								}
								else{
									lon_ptr.pop_back();
								}
							}
						}
					}

				}
				pch = strtok_r (NULL, "[", &end_str);
				wordNo++;

			}
			//printf("num lonlat were %d x 2\n",lonlatno);
		}
		lineNo++;
		if(lonlatno<=0){
			lineNo--;
		}
		else if(lineNo>0){
			PTI.push_back(PTI[lineNo-1] + lonlatno);
		}

		if(PTI[lineNo]>=max_pts_limit){
			break;
		}
		//printf("Line %d\n",lineNo);
	}
	fclose ( fp_data );

	int num_points = lon_ptr.size();
	PTI.pop_back();
//	printf("Number of trajectories are %d\n",PTI.size());
//	printf("Number of points is %d\n",lon_ptr.size());
//
//	printf("Extending the dataset now\n");
//	srand(time(NULL));
//	int num_traj = PTI.size();
//	PTI.resize(num_traj*repeat);
//	lon_ptr.resize(num_points*repeat);
//	lat_ptr.resize(num_points*repeat);
//
//	float randOffset = 10;
//	for(int i=1;i<repeat;i++){
//		for(int j=0;j<num_points;j++){
//			lon_ptr[j+num_points*(i)] = lon_ptr[j] + ((double) rand() / (RAND_MAX))*randOffset;
//			lat_ptr[j+num_points*(i)] = lat_ptr[j] + ((double) rand() / (RAND_MAX))*randOffset;
//		}
//
//		PTI[num_traj*(i)] = PTI[num_traj*(i)-1]+PTI[0];
//		for(int j=1;j<num_traj;j++){
//			PTI[j+num_traj*(i)] = PTI[j+num_traj*(i)-1] + (PTI[j] - PTI[j-1]);
//		}
//
//	}
//
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

int read_data_traj_per_line_3d(char* filename, int max_pts_limit, traj_data_t** data, int repeat)
{

	//This function reads and segments trajectories in dataset in the following format:
	//The first line indicates number of variables per point (I'm ignoring that and assuming 2)
	//The second line indicates total trajectories in file (I'm ignoring that and
	//observing how many are there by reading them). All lines that follow contains a
	//trajectory separated by new line.
	//The first number in the trajectory is the number of points followed by
	//location points separated by spaces

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
//	printf("Number of trajectories are %d\n",PTI.size());
//	printf("Number of points is %d\n",lon_ptr.size());
//
//	printf("Extending the dataset now\n");
//	srand(time(NULL));
//	int num_traj = PTI.size();
//	PTI.resize(num_traj*repeat);
//	lon_ptr.resize(num_points*repeat);
//	lat_ptr.resize(num_points*repeat);
//
//	float randOffset = 10;
//	for(int i=1;i<repeat;i++){
//		for(int j=0;j<num_points;j++){
//			lon_ptr[j+num_points*(i)] = lon_ptr[j] + ((double) rand() / (RAND_MAX))*randOffset;
//			lat_ptr[j+num_points*(i)] = lat_ptr[j] + ((double) rand() / (RAND_MAX))*randOffset;
//		}
//
//		PTI[num_traj*(i)] = PTI[num_traj*(i)-1]+PTI[0];
//		for(int j=1;j<num_traj;j++){
//			PTI[j+num_traj*(i)] = PTI[j+num_traj*(i)-1] + (PTI[j] - PTI[j-1]);
//		}
//
//	}
//
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

int read_data_traj_per_line_Ngsim(char* filename, int max_pts_limit, traj_data_t** data, int repeat)
{

	//This function reads and segments trajectories in dataset in the following format:
	//The first line indicates number of variables per point (I'm ignoring that and assuming 2)
	//The second line indicates total trajectories in file (I'm ignoring that and
	//observing how many are there by reading them). All lines that follow contains a
	//trajectory separated by new line.
	//The first number in the trajectory is the number of points followed by
	//location points separated by spaces

    ifstream file ( filename);

    int lineNo = 0;
    int wordNo = 0;
    int lonlatno = 100;
    bool badLongitude = false;

    int trajID=0;
    int ptsInTraj=0;

    PTI.push_back(0);
    TKI.push_back(0);

    string thisWord;
    bool firstLine = false;
    int ptCt = 0;
	while (file.good()) {
		if(ptCt<1000000){
			getline ( file, thisWord, ',' );
			lon_ptr.push_back(stof(thisWord));
			getline ( file, thisWord);
			lat_ptr.push_back(stof(thisWord));
			ptCt++;
		}
		else break;
	}
//	lon_ptr.pop_back();
//	lat_ptr.pop_back();

	printf("lon %f lat %f\nlon %f lat %f\nlon %f lat %f\n",lon_ptr[0],lat_ptr[0],lon_ptr[1],lat_ptr[1],lon_ptr[2],lat_ptr[2]);

	int num_points = lon_ptr.size();
	PTI.pop_back();
//	printf("Number of trajectories are %d\n",PTI.size());
//	printf("Number of points is %d\n",lon_ptr.size());
//
//	printf("Extending the dataset now\n");
//	srand(time(NULL));
//	int num_traj = PTI.size();
//	PTI.resize(num_traj*repeat);
//	lon_ptr.resize(num_points*repeat);
//	lat_ptr.resize(num_points*repeat);
//
//	float randOffset = 10;
//	for(int i=1;i<repeat;i++){
//		for(int j=0;j<num_points;j++){
//			lon_ptr[j+num_points*(i)] = lon_ptr[j] + ((double) rand() / (RAND_MAX))*randOffset;
//			lat_ptr[j+num_points*(i)] = lat_ptr[j] + ((double) rand() / (RAND_MAX))*randOffset;
//		}
//
//		PTI[num_traj*(i)] = PTI[num_traj*(i)-1]+PTI[0];
//		for(int j=1;j<num_traj;j++){
//			PTI[j+num_traj*(i)] = PTI[j+num_traj*(i)-1] + (PTI[j] - PTI[j-1]);
//		}
//
//	}
//
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


int preprocess_data(char* filename, traj_data_t** data, int max_pts_limit, int repeat)
{
    vector<Point> h_points;

    //Here we choose the appropriate parser for our dataset
    //This basically tests the filename and runs the correct routine

    int ret = -1;
    char *rest = (char*)malloc(sizeof(char) * 254);;
    strcpy(rest,filename);
    char *token, *dataSetName;
    while ((token = strtok_r(rest, "/", &rest))){
            dataSetName = token;
    }

    if(strcmp(dataSetName,"3D_spatial_network.txt")==0){
    printf("[INFO] Spatial 3D Data detected!!!!\n");
    int num_points = read_data_traj_per_line_3d(filename, max_pts_limit, data, repeat);
    ret = 0;
  } else if(strcmp(dataSetName,"iono_20min_2Mpts_3D.txt")==0){
    printf("[INFO] IONO 3D Data detected!!!!\n");
    int num_points = read_data_traj_per_line_3d(filename, max_pts_limit, data, repeat);
    ret = 0;
  } else {
    printf("[INFO] Using 2D dataset.\n");
    int num_points =
        read_data_traj_per_line(filename, max_pts_limit, data, repeat);
    ret = 0;
  }

    return ret;

    // for(int i = 0; i < (*data)->TKI_length + 1; i++) {
    //     printf("TKI[%d] = %d\n", i, (*data)->TKI[i]);
    // }
    // for(int i = 0; i < (*data)->PTI_length + 1; i++) {
    //     printf("PTI[%d] = %d\n", i, (*data)->PTI[i]);
    // }
    // for(int i = 0; i < (*data)->num_tracks; i++) {
    //     printf("TTT[%d] = %d\n", i, (*data)->track_to_traj[i]);
    // }
}
