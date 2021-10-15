#include "makeGraph.h"

void makeGraph(int NUM_BLOCKS, int BLOCK_THREADS,float *x, float *y, int numPoints, int minPts, float R, Graph* distGraph, bool** clusterType, report_t *report){

    //Initialize memory for all the arrays
	long unsigned int *dNodes;
	int  *dEdges;
    bool *dClusterType;
    float *d_x, *d_y;

    //hAdjMatrix = (bool*) malloc(sizeof(bool)*(numPoints*numPoints));
    gpuErrchk(cudaMalloc((void**)&dNodes, sizeof(long unsigned int) * (numPoints +1)));
    gpuErrchk(cudaMalloc((void**)&d_x, sizeof(float) * numPoints));
    gpuErrchk(cudaMalloc((void**)&d_y, sizeof(float) * numPoints));
    gpuErrchk(cudaMalloc((void**)&dClusterType, sizeof(bool) * numPoints));

    //Copy stuff into the device
    gpuErrchk(cudaMemcpy(d_x, x, sizeof(float) * numPoints, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, sizeof(float) * numPoints, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dClusterType, *clusterType, sizeof(bool) * numPoints, cudaMemcpyHostToDevice));

    //Make Adjacency Matrix of all points within radius of all points
    dim3 dimGrid(NUM_BLOCKS,1);
    dim3 dimBlock(BLOCK_THREADS,1);
    fillNodes<<<dimGrid,dimBlock>>>(minPts,R,numPoints,d_x,d_y,dNodes,dClusterType);

    //Get back the info on invididual points (Core or Border)
	gpuErrchk(cudaMemcpy(*clusterType, dClusterType, sizeof(bool) * numPoints, cudaMemcpyDeviceToHost));
    cudaFree(dClusterType);

    //Prints for debugging and monitoring
	if(PRINT_LOG){
		int* hNodes = (int*) malloc(sizeof(int)*(numPoints+1));
		gpuErrchk(cudaMemcpy(hNodes, dNodes, sizeof(int) * (numPoints+1), cudaMemcpyDeviceToHost));
		printf("Cluster type ");
		for(int i=0;i<numPoints;i++){
			printf("%d ",(*clusterType)[i]);
		}
		printf("\n");
		for(int i=0;i<numPoints;i++){
				   printf("%d " ,hNodes[i]);
		 }
		printf("\nAdj Matrix:");
		free(hNodes);
	}

	//CPU exclusive scan because Thrust was being difficult with large data
//	hNodes[numPoints] = 0;
//	int temp = hNodes[0];
//	int temp2 = hNodes[1];
//	hNodes[0] = 0;
//	hNodes[1] = temp;
//	for(int i=2;i<=numPoints;i++){
//		temp = hNodes[i];
//		hNodes[i] += temp2 + hNodes[i-1];
//		temp2 = temp;
//	}
	//CPU exclusive scan


	//Use Thrust's exclusive scan to make array with neighboring points
    thrust::device_ptr<long unsigned int> in_ptr(dNodes);
    thrust::exclusive_scan(in_ptr, in_ptr + (numPoints + 1), in_ptr);

    //"Get the last element of exclusive scan to allocate memory
    long unsigned int *totalEdges = (long unsigned int*)malloc(sizeof(long unsigned int));
    gpuErrchk(cudaMemcpy(totalEdges, &dNodes[numPoints], sizeof(long unsigned int), cudaMemcpyDeviceToHost));

    printf("Total edges are %lu\n",(*totalEdges));

    //Save memory by allocating just what is needed
    gpuErrchk(cudaMalloc((void**)&dEdges, sizeof(int) * (*totalEdges)));

    //Capture memory info since this is where its consumed most
    size_t free_byte, total_byte ;
	cudaMemGetInfo( &free_byte, &total_byte) ;
	set_finalMemory(report,((double)total_byte - (double)free_byte)/1024.0/1024.0);

    //Prints for debugging and monitoring
    if(PRINT_LOG)printf("\nWe have a total of %d edges",*totalEdges);

    //Get adjacency list in contiguous, memory efficient, integer form
    fillEdges<<<dimGrid,dimBlock>>>(numPoints, R,d_x,d_y, dNodes,dEdges);

    //Send the data back
    distGraph->edges = dEdges;
    distGraph->nodes = dNodes;
    distGraph->totalEdges = *totalEdges;

    //Clean up the mess
    cudaFree(d_x);
    cudaFree(d_y);

}

__global__ void fillNodes(int minPts, float R, int numPoints, float *d_x,
								float *d_y,long unsigned int * dNodes, bool* dClusterType){
    //Grid Stride Loop
    for (int tID = blockIdx.x * blockDim.x + threadIdx.x; 
         tID < numPoints;
         tID += blockDim.x * gridDim.x){

        float thisPointX = d_x[tID];
        float thisPointY = d_y[tID];
        float distance;
        dNodes[tID] = 0;
        __syncthreads();
        for(int i=0; i<numPoints; i++){
            distance = euclidean_distance(thisPointX,thisPointY,d_x[i],d_y[i]);
            if(distance <= R){
                dNodes[tID]++;
            }
        }
        __syncthreads();

        if(dNodes[tID]>=minPts){
        	dClusterType[tID] = Core;
        }
    }
}


__global__ void fillEdges(int numPoints, float R,float *d_x,float *d_y, long unsigned int * dNodes,int *dEdges){
	for (int tID = blockIdx.x * blockDim.x + threadIdx.x;
		          tID < numPoints;
		          tID += blockDim.x * gridDim.x)
	{
        float thisPointX = d_x[tID];
        float thisPointY = d_y[tID];
        float distance;
		int edgeOffset = dNodes[tID];
		__syncthreads();
		for(int i=0; i<numPoints; i++){
			distance = euclidean_distance(thisPointX,thisPointY,d_x[i],d_y[i]);
			if(distance <= R){
				dEdges[edgeOffset] = i;
				edgeOffset++;
			}
		}
	}

}
__device__ __host__
float euclidean_distance(float p1_x, float p1_y,
                         float p2_x, float p2_y)
{
    float x_diff = p1_x - p2_x;
    float y_diff = p1_y - p2_y;
    return sqrt(x_diff * x_diff + y_diff * y_diff);
}
