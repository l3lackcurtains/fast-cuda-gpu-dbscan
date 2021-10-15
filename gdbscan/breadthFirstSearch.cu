#include "breadthFirstSearch.h"


void identifyCluster(int NUM_NODES, int NUM_BLOCKS, int BLOCK_THREADS, long unsigned int* Va, int* Ea, int** clusterIDs, bool** clusterType, int* numClusters){
	//Set the first cluster ID
	int thisClusterID = Not_Visited+1;

	//Visit every node
	for(int i=0; i<NUM_NODES; i++){
		if((*clusterIDs)[i]==Not_Visited && (*clusterType)[i]==Core){
			(*clusterIDs)[i] = thisClusterID;

			//Run BFS on the core point
			BreadthFirstSearch(NUM_NODES,NUM_BLOCKS,BLOCK_THREADS,i,Va, Ea, *clusterIDs, *clusterType, thisClusterID);
			thisClusterID++;

			//Prints for debugging and monitoring
			if(PRINT_LOG){
				printf("\nClusterIDs: ");
				for(int i=0;i<NUM_NODES;i++){
					printf("%d ",(*clusterIDs)[i]);
				}
			printf("\n");
			}
		}

	}
	*numClusters = thisClusterID;
	//Clean up the mess
	cudaFree(Va);
	cudaFree(Ea);
}

void BreadthFirstSearch(int NUM_NODES, int NUM_BLOCKS, int BLOCK_THREADS, int source, long unsigned int* Va, int* Ea, int* clusterIDs, bool* clusterType, int thisClusterID)
{
	//Initialize frontiers and visited arrays
	bool *frontier = (bool*)malloc(sizeof(bool)*NUM_NODES);
	bool *visited = (bool*)malloc(sizeof(bool)*NUM_NODES);
	int *cost = (int*)malloc(sizeof(int)*NUM_NODES);
	for(int i=0;i<NUM_NODES;i++){
		frontier[i] = false;
		visited[i] = false;
		cost[i] = 0;
	}

	//Set the source as a frontier
	frontier[source] = true;

	//Variables used by the kernel
	bool* Fa;
	cudaMalloc((void**)&Fa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Fa, frontier, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);
	bool* Xa;
	cudaMalloc((void**)&Xa, sizeof(bool)*NUM_NODES);
	cudaMemcpy(Xa, visited, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);
	int* Ca;
	cudaMalloc((void**)&Ca, sizeof(int)*NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int)*NUM_NODES, cudaMemcpyHostToDevice);
	bool* dClusterType;
	cudaMalloc((void**)&dClusterType, sizeof(bool)*NUM_NODES);
	cudaMemcpy(dClusterType, clusterType, sizeof(bool)*NUM_NODES, cudaMemcpyHostToDevice);
	bool done;
	bool* d_done;
	cudaMalloc((void**)&d_done, sizeof(bool));
	int count = 0;

	dim3 dimGrid(NUM_BLOCKS,1);
	dim3 dimBlock(BLOCK_THREADS,1);

	//Run kernel until all frontiers are explored
	do{
		count++;
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		BreadthFirstSearchKernel <<<dimGrid,dimBlock>>>(NUM_NODES, Va, Ea, Fa, Xa, Ca,dClusterType,d_done);
		cudaMemcpy(&done, d_done , sizeof(bool), cudaMemcpyDeviceToHost);

	}
	while(!done);

	//Get the visited data back
	cudaMemcpy(cost, Ca, sizeof(int)*NUM_NODES, cudaMemcpyDeviceToHost);
	cudaMemcpy(visited, Xa, sizeof(bool)*NUM_NODES, cudaMemcpyDeviceToHost);

	//Prints for debugging and monitoring
	if(PRINT_LOG){
		printf("\nNumber of times the kernel is called : %d \n", count);
		printf("\Visited: ");
		for (int i = 0; i<NUM_NODES; i++)
				printf( "%d    ", visited[i]);
			printf("\n");
	}

	//Set all visited nodes to this cluster ID
	for(int i=0;i<NUM_NODES;i++){
		if(visited[i]){
			clusterIDs[i] = thisClusterID;
		}
	}

	//Clean up the mess
	cudaFree(Fa);
	cudaFree(Xa);
	cudaFree(Ca);
	cudaFree(dClusterType);
	cudaFree(d_done);
	free(frontier);
	free(visited);
	free(cost);
}



__global__ void BreadthFirstSearchKernel(int NUM_NODES, long unsigned  *Va, int *Ea, bool *Fa, bool *Xa, int *Ca,bool *dClusterType, bool *done)
{

	for (int id = blockIdx.x * blockDim.x + threadIdx.x;
	         id < NUM_NODES;
	         id += blockDim.x * gridDim.x){


	if (Fa[id] == true && Xa[id] == false)
	{

		//printf("%d ", id); //This printf gives the order of vertices in BFS
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads();
		if(dClusterType[id] == Core){
			int k = 0;
			int i;
			int start = Va[id];
			int end = Va[id+1];
			for (int i = start; i < end; i++)
			{
				int nid = Ea[i];

				if (Xa[nid] == false)
				{
					Ca[nid] = Ca[id] + 1;
					Fa[nid] = true;
					*done = false;
				}

			}

		}

	}

//	if (blockIdx.x * blockDim.x + threadIdx.x > NUM_NODES)
//		*done = false;

	}
}
