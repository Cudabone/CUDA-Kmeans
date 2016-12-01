#include <stdio.h>
#include <assert.h>

void cudaErrorCheck(cudaError_t err, const char *s);
__device__ inline float euclidean_dist(float *d_Clusters, 
										float *d_Objects, 
										int numClusters,
										int numObjects,
										int numCoords,
										int objectNum,
										int clusterNum)
{
	float dist = 0.0;

	int i;
	int objIdx,clusterIdx;
	for(i = 0; i < numCoords; i++)
	{
		//Index for object and cluster
		objIdx = numObjects*i + objectNum;
		clusterIdx = numClusters*i + clusterNum;

		dist += (d_Objects[objIdx] - d_Clusters[clusterIdx]) * 
			(d_Objects[objIdx]- d_Clusters[clusterIdx]);
	}
	return dist;
}
__global__ void find_nearest_cluster(float *d_Clusters,
										float *d_Objects,
										float *d_Deltas,
										int *d_Membership,
										int numClusters,
										int numObjects,
										int numCoords)
{
	//Shared to store delta's for block
	extern __shared__ float delta[]; // 
	//Thread index into objects
	unsigned int objectNum = blockDim.x * blockIdx.x + threadIdx.x;

	//Ensure that thread within # objects and block has a cluster
	if(objectNum >= numObjects)
		return;

	//Set deltas to 0 and sync threads
	delta[threadIdx.x] = 0.0;
	__syncthreads();

	//find distance
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist =
		euclidean_dist(d_Clusters,d_Objects,numClusters,numObjects,numCoords,objectNum,0);

    for (i=1; i<numClusters; i++) {
        dist = 
		euclidean_dist(d_Clusters,d_Objects,numClusters,numObjects,numCoords,objectNum,i);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
	//Each thread now knows the closest cluster to its object
	if(d_Membership[objectNum] != index)
		delta[threadIdx.x] += 1.0;

	//TODO Can use shared mem for this if better
	//Assign Membership for Object
	d_Membership[objectNum] = index;

	//Reduce Deltas for block
	__syncthreads();
	for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		if(threadIdx.x < stride)
			delta[threadIdx.x] += delta[threadIdx.x + stride];  
		__syncthreads();
	}

	//Assign delta for block in device deltas
	if(threadIdx.x == 0)
		d_Deltas[blockIdx.x] = delta[0];
}
/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
int cuda_kmeans(float **objects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int    *membership,   /* out: [numObjs] */
               float **clusters)     /* out: [numClusters][numCoords] */
{
	float *d_Clusters;
	float *d_Objects;
	float *h_Deltas;
	float *d_Deltas;
	int *d_Membership;
	float **newClusters;
	int *newClusterSize;

	//float *h_Clusters;
	//float *h_Objects;
	//h_Clusters = (float *)malloc(numClusters*numCoords*sizeof(float));
	//h_Objects = (float *)malloc(numObjs*numCoords*sizeof(float));

	int i,j;
	/*
	//Flatten clusters and objects for transfer
	for(i = 0; i < numClusters; i++)
	{
		for(j = 0; j < numCoords;j++)
		{
			h_Clusters[j*numClusters + i] = clusters[i][j];
		}
	}
	for(i = 0; i < numObjs; i++)
	{
		for(j = 0; j < numCoords;j++)
		{
			h_Objects[j*numObjs + i] = objects[i][j];
		}
	}
	*/

	//Number of threads per block
	int blocksize = 32;
	int numblocks = ceil((float)numObjs/blocksize);

	h_Deltas = (float *)malloc(numblocks*sizeof(float));
	//Allocate and Initialize newClusterSize to 0's
	newClusterSize = (int *)calloc(numClusters,sizeof(int));

	//Create new clusters as is done in original algorithm
    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);

	//Initialize all pointers for newClusters
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

	cudaErrorCheck(cudaMalloc((void
					**)&d_Clusters,numClusters*numCoords*sizeof(float)),
				"CMalloc d_Clusters");
	cudaErrorCheck(cudaMalloc((void
					**)&d_Objects,numObjs*numCoords*sizeof(float)),
			"Cmalloc d_Objects");
	cudaErrorCheck(cudaMalloc((void
					**)&d_Deltas,numblocks*sizeof(float)),
			"Cmalloc d_Deltas");
	cudaErrorCheck(cudaMalloc((void
					**)&d_Membership,numObjs*sizeof(int)),
			"Cmalloc d_Membership");
	//cudaErrorCheck(cudaMemcpy2DToArray(,cudaMemcpyHostToDevice))
	int loop = 0;
	int index;
	float delta;

	// Initialize Membership
	for(i = 0; i < numObjs; i++)
		membership[i] = -1;

	//Copy all host data to device
	cudaErrorCheck(cudaMemcpy((void *)d_Membership,(const void
					*)membership,numObjs*sizeof(int),cudaMemcpyHostToDevice),"Members to device");

	cudaErrorCheck(cudaMemcpy((void *)d_Objects,(const void *)objects[0],numObjs*numCoords*sizeof(float),cudaMemcpyHostToDevice),
			"Objects to Device");

	do{
		//copy initial/new cluster centers to device 
	cudaErrorCheck(cudaMemcpy((void *)d_Clusters,(const void
					*)clusters[0],numClusters*numCoords*sizeof(float),cudaMemcpyHostToDevice),
			"Clusters to Device");
		delta = 0.0;
		//TODO Can pretty much implement this entire loop i think in CUDA only
		//find_nearest_cluster
		cudaDeviceSynchronize();
		/* find the array index of nestest cluster center */
		//TODO Create and Replace find_nearest cluster for Cuda

		//index = find_nearest_cluster(numClusters, numCoords, objects[i],clusters);
		find_nearest_cluster<<<numblocks,blocksize,blocksize>>>
			(d_Clusters,d_Objects,d_Deltas,d_Membership,numClusters,numObjs,numCoords);
		cudaDeviceSynchronize();

		cudaErrorCheck(cudaMemcpy((void *)h_Deltas,(const void
						*)d_Deltas,numblocks*sizeof(float),cudaMemcpyDeviceToHost),
				"Copy deltas to host");
		//Sum all deltas from each block
		for(i = 0; i < numblocks; i++)
			delta += h_Deltas[i];

		//The Rest is mostly left unchanged

		/* update new cluster center : sum of objects located within */
		for(i = 0; i < numObjs; i++)
		{
			newClusterSize[index]++;
			for (j=0; j<numCoords; j++)
				newClusters[index][j] += objects[i][j];
		}
		/* average the sum and replace old cluster center with newClusters */
		for (i=0; i<numClusters; i++) {
			for (j=0; j<numCoords; j++) {
				if (newClusterSize[i] > 0)
					clusters[i][j] = newClusters[i][j] / newClusterSize[i];
				newClusters[i][j] = 0.0;   /* set back to 0 */
			}
			newClusterSize[i] = 0;   /* set back to 0 */
		}

		delta /= numObjs;
	} while (delta > threshold && loop++ < 500);

	//Copy membership back
	cudaErrorCheck(cudaMemcpy((void *)membership,(const void
					*)d_Membership,numObjs*sizeof(int),cudaMemcpyDeviceToHost),
			"Membership to Host");

	//Free All Cuda and C Pointers

	cudaFree(d_Clusters);	
	cudaFree(d_Objects);
	cudaFree(d_Deltas);
	cudaFree(d_Membership);

	free(h_Deltas);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

	return 1;
}
void cudaErrorCheck(cudaError_t err, const char *s)
{
	if(err != cudaSuccess)
	{
		printf("%s error: %s\n",s,cudaGetErrorString(err));
		exit(0);
	}
}
