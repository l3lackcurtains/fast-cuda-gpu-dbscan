#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "makeGraph.h"
#include "breadthFirstSearch.h"
#include "read_dataset.h"
#include "report.h"

#define PORTO 0
#define SPATIAL 0
#define NGSI 1
#define IONO2D 0

int NUM_NODES;
double RADIUS;
int MIN_POINTS;
int BLOCK_THREADS;
int NUM_BLOCKS;


#define NUM_REPORTS 3000000
report_t report[NUM_REPORTS];
int current=0;
bool readOnce = false;
traj_data_t* d_data;
traj_data_t* h_data;


void read_dataset(char* filename, traj_data_t** d_data,
                  traj_data_t** h_data, int max_pts_limit)
{

#ifdef MEMORY_PRINT
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;
    printf("Before reading data\n");
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            ((double)total_byte - (double)free_byte)/1024.0/1024.0,
            ((double)free_byte)/1024.0/1024.0, ((double)total_byte)/1024.0/1024.0);
#endif

    //    *d_data = (traj_data_t*) malloc(sizeof(traj_data_t) * 1);
    //cudaMalloc((void**)d_data, sizeof(traj_data_t) * 1);

    *h_data = (traj_data_t*) malloc(sizeof(traj_data_t) * 1);

    preprocess_data(filename, h_data, max_pts_limit, 1);

    //move_data_to_gpu(h_data, d_data);

#ifdef MEMORY_PRINT
    cudaMemGetInfo( &free_byte, &total_byte ) ;
    printf("After reading data\n");
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
            ((double)total_byte - (double)free_byte)/1024.0/1024.0,
            ((double)free_byte)/1024.0/1024.0, ((double)total_byte)/1024.0/1024.0);
#endif

}

void G_DBSCAN(traj_data_t* h_data,int **clusterIDs,bool **clusterType, int * numClusters){

	//Initialize cluster set
	for(int i=0;i<NUM_NODES;i++){
		(*clusterIDs)[i] = Not_Visited;
		(*clusterType)[i] = Border;
	}

	Graph* distGraph = (Graph*)malloc(sizeof(Graph));

	//Create adjacency matrix and an edge between all connected nodes
	makeGraph(NUM_BLOCKS,BLOCK_THREADS,h_data->x, h_data->y, NUM_NODES, MIN_POINTS, RADIUS, distGraph, clusterType, &report[current]);

	//Do breadth first search to find clusters in the graph
	identifyCluster(NUM_NODES,NUM_BLOCKS,BLOCK_THREADS, distGraph->nodes, distGraph->edges, clusterIDs, clusterType, numClusters);

	//Clean up the mess
	free(distGraph);
}

void runTest(){
	for(int i=0;i<REPEAT_BENCHMARK;i++){
		//Get initial benchmarks
		set_epsilon(&report[current],RADIUS);
		set_minPts(&report[current],MIN_POINTS);
		set_num_points(&report[current],NUM_NODES);
		set_num_threads(&report[current],BLOCK_THREADS);
		set_num_blocks(&report[current],NUM_BLOCKS);
		gettimeofday(&(report[current].total_time_begin), NULL);
		gettimeofday(&(report[current].data_read_time_begin), NULL);

		//Read the dataset
		if(!readOnce){
			int max_pts_limit = 400000;
			d_data = NULL;
			h_data = NULL;
			char* filename;
			if(PORTO) filename = "/data/dbscan/Porto_taxi_data.csv";
			if(SPATIAL) filename = "/data/dbscan/3D_spatial_network.txt";
			if(NGSI) filename = "/data/dbscan/NGSIM_Data.txt";
			if(IONO2D) filename = "/data/geodata/iono_20min_2Mpts_2D.txt";
			read_dataset(filename, &d_data, &h_data, max_pts_limit);
			readOnce = true;
		}
		size_t free_byte, total_byte;
		cudaMemGetInfo( &free_byte, &total_byte) ;
		set_initMemory(&report[current],((double)total_byte - (double)free_byte)/1024.0/1024.0);
		gettimeofday(&(report[current].data_read_time_end), NULL);
		gettimeofday(&(report[current].processing_time_begin), NULL);

		//Initialize output of the program
		int *clusterIDs = (int*)malloc(sizeof(int)*NUM_NODES);
		bool *clusterType = (bool*)malloc(sizeof(bool)*NUM_NODES);
		int numClusters;

		//Run the G-DBScan Algorithm
		G_DBSCAN(h_data, &clusterIDs, &clusterType, &numClusters);

		//Get final benchmarks
		gettimeofday(&(report[current].processing_time_end), NULL);
		gettimeofday(&(report[current].total_time_end), NULL);
		set_num_clusters(&report[current],numClusters);
		print_report(&report[current]);
		current++;

		
		//Clean up the mess
		free(clusterIDs);
		free(clusterType);
	}
}

int main()
{
	init_csv();
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
	
		defaultMin = 4;
		defaultR = 1.5;
	
		defaultPts = 400000;
	 }

	RADIUS = defaultRStress;
	MIN_POINTS = defaultMinStress;
	NUM_NODES = defaultPts;
	BLOCK_THREADS = 256;
	NUM_BLOCKS = (NUM_NODES/BLOCK_THREADS) +1;
	// runTest();

	
	// //Test impact of radius
	// for(int i=0;i<rangeSize;i++){
	//   RADIUS = setOfR[i];
	//   MIN_POINTS = defaultMin;
	//   NUM_NODES = defaultPts;
	// 	BLOCK_THREADS = 256;
	// 	NUM_BLOCKS = (NUM_NODES/BLOCK_THREADS) +1;
	// 	runTest();
	// }

	// //Test impact of minPts
	// for(int i=0;i<rangeSize;i++){
	// 	RADIUS = defaultR;
	// 	MIN_POINTS = setOfMinPts[i];
	// 	NUM_NODES = defaultPts;
	// 	  BLOCK_THREADS = 256;
	// 	  NUM_BLOCKS = (NUM_NODES/BLOCK_THREADS) +1;
	// 	  runTest();
	//   }

//	//Test impact of points
	for(int i=0;i<rangeSize;i++){
	  RADIUS = defaultR;
	  MIN_POINTS = defaultMin;
	  NUM_NODES = setOfDataSize[i];
		BLOCK_THREADS = 256;
		NUM_BLOCKS = (NUM_NODES/BLOCK_THREADS) +1;
		runTest();
	}


}
