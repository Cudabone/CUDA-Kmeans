/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_main.c   (an sequential version)                      */
/*   Description:  This program shows an example on how to call a subroutine */
/*                 that implements a simple k-means clustering algorithm     */
/*                 based on Euclid distance.                                 */
/*   Input file format:                                                      */
/*                 ascii  file: each line contains 1 data object             */
/*                 binary file: first 4-byte integer is the number of data   */
/*                 objects and 2nd integer is the no. of features (or        */
/*                 coordinates) of each object                               */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* 
Copyright (c) 2005 Northwertern University
All rights reserved.

Access and use of this software shall impose the following obligations and
understandings on the user. The user is granted the right, without any fee or
cost, to use, copy, modify, alter, enhance and distribute this software, and
any derivative works thereof, and its supporting documentation for any purpose
whatsoever, provided that this entire notice appears in all copies of the
software, derivative works and supporting documentation.  Further, Northwestern
University requests that the user credit Northwestern University in any
publications that result from the use of this software or in any product that
includes this software.  The name Northwestern University, however, may not be
used in any advertising or publicity to endorse or promote any products or
commercial entity unless specific written permission is obtained from
Northwestern University. The user also understands that Northwestern University
is not obligated to provide the user with any support, consulting, training or
assistance of any kind with regard to the use, operation and performance of
this software nor to provide the user with any updates, revisions, new versions
or "bug fixes."

THIS SOFTWARE IS PROVIDED BY NORTHWESTERN UNIVERSITY "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL NORTHWESTERN UNIVERSITY BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE,
DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
ACTION, ARISING OUT OF OR IN CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF
THIS SOFTWARE.

cuda_main.cu 
Modified from seq_main.c 
Author: Matt Mikuta
Illinois Tech
*/

#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <errno.h>

int      _debug;
#include "kmeans.h"

/*---< usage() >------------------------------------------------------------*/
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -c centers     : file containing initial centers. default: filename\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n"
        "       -q             : quiet mode\n"
        "       -d             : enable debug mode\n"
        "       -h             : print this help information\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j, isBinaryFile, is_output_timing, verbose;

           int     numClusters, numCoords, numObjs;
           int    *membership;    /* [numObjs] */
           char   *filename, *center_filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
           double  timing, io_timing, clustering_timing;

    /* some default values */
    _debug           = 0;
    verbose          = 1;
    threshold        = 0.001;
    numClusters      = 0;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;
    center_filename  = NULL;

    while ( (opt=getopt(argc,argv,"p:i:c:n:t:abdohq"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'c': center_filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case 'q': verbose = 0;
                      break;
            case 'd': _debug = 1;
                      break;
            case 'h':
            default: usage(argv[0], threshold);
                      break;
        }
    }
    if (center_filename == NULL)
        center_filename = filename;

    if (filename == 0 || numClusters <= 1) usage(argv[0], threshold);

    if (is_output_timing) io_timing = wtime();

    /* read data points from file ------------------------------------------*/
    printf("reading data points from file %s\n",filename);

    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);

    if (numObjs < numClusters) {
        printf("Error: number of clusters must be larger than the number of data points to be clustered.\n");
        free(objects[0]);
        free(objects);
        return 1;
    }

    /* allocate a 2D space for clusters[] (coordinates of cluster centers)
       this array should be the same across all processes                  */
    clusters    = (float**) malloc(numClusters *             sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    /* read the first numClusters elements from file center_filename as the
     * initial cluster centers*/
    if (center_filename != filename) {
        printf("reading initial %d centers from file %s\n", numClusters,
               center_filename);
        /* read the first numClusters data points from file */
        read_n_objects(isBinaryFile, center_filename, numClusters,
                       numCoords, clusters);
    }
    else {
        printf("selecting the first %d elements as initial centers\n",
               numClusters);
        /* copy the first numClusters elements in feature[] */
        for (i=0; i<numClusters; i++)
            for (j=0; j<numCoords; j++)
                clusters[i][j] = objects[i][j];
    }

    /* check initial cluster centers for repeatition */
	/*
    if (check_repeated_clusters(numClusters, numCoords, clusters) == 0) {
        printf("Error: some initial clusters are repeated. Please select distinct initial centers\n");
        return 1;
    }
	*/

    if (_debug) {
        printf("Sorted initial cluster centers:\n");
        for (i=0; i<numClusters; i++) {
            printf("clusters[%d]=",i);
            for (j=0; j<numCoords; j++)
                printf(" %6.2f", clusters[i][j]);
            printf("\n");
        }
    }

    if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }

    /* start the timer for the core computation -----------------------------*/
    /* membership: the cluster id for each data object */
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

	cuda_kmeans(objects,numCoords,numObjs,numClusters,threshold,membership,clusters);

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
    }

    /* output: the coordinates of the cluster centres ----------------------*/
    file_write(filename, numClusters, numObjs, numCoords, clusters,
               membership, verbose);

    free(membership);
    free(clusters[0]);
    free(clusters);

    /*---- output performance numbers ---------------------------------------*/
    if (is_output_timing) {
        io_timing += wtime() - timing;
        printf("\nPerforming **** Regular Kmeans (sequential version) ****\n");

        printf("Input file:     %s\n", filename);
        printf("numObjs       = %d\n", numObjs);
        printf("numCoords     = %d\n", numCoords);
        printf("numClusters   = %d\n", numClusters);
        printf("threshold     = %.4f\n", threshold);

        printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", clustering_timing);
    }

    return(0);
}

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
	/*
	__syncthreads();
	for(unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		if(threadIdx.x < stride)
			delta[threadIdx.x] += delta[threadIdx.x + stride];  
		__syncthreads();
	}
	*/

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
	int blocksize = 1024;
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
		find_nearest_cluster<<<numblocks,blocksize,blocksize*sizeof(float)>>>
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
#define MAX_CHAR_PER_LINE 128


/*---< file_read() >---------------------------------------------------------*/
float** file_read(int   isBinaryFile,  /* flag: 0 or 1 */
                  char *filename,      /* input file name */
                  int  *numObjs,       /* no. data objects (local) */
                  int  *numCoords)     /* no. coordinates */
{
    float **objects;
    int     i, j, len;
    ssize_t numBytesRead;

    if (isBinaryFile) {  /* input file is in raw binary format -------------*/
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
        numBytesRead = read(infile, numObjs,    sizeof(int));
        assert(numBytesRead == sizeof(int));
        numBytesRead = read(infile, numCoords, sizeof(int));
        assert(numBytesRead == sizeof(int));
        if (_debug) {
            printf("File %s numObjs   = %d\n",filename,*numObjs);
            printf("File %s numCoords = %d\n",filename,*numCoords);
        }

        /* allocate space for objects[][] and read all objects */
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        numBytesRead = read(infile, objects[0], len*sizeof(float));
        assert(numBytesRead == len*sizeof(float));

        close(infile);
    }
    else {  /* input file is in ASCII format -------------------------------*/
        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }

        /* first find the number of objects */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        (*numObjs) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            /* check each line to find the max line length */
            while (strlen(line) == lineLen-1) {
                /* this line read is not complete */
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                /* increase lineLen */
                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }

            if (strtok(line, " \t\n") != 0)
                (*numObjs)++;
        }
        rewind(infile);
        if (_debug) printf("lineLen = %d\n",lineLen);

        /* find the no. coordinates of each object */
        (*numCoords) = 0;
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") != 0) {
                /* ignore the id (first coordiinate): numCoords = 1; */
                while (strtok(NULL, " ,\t\n") != NULL) (*numCoords)++;
                break; /* this makes read from 1st object */
            }
        }
        rewind(infile);
        if (_debug) {
            printf("File %s numObjs   = %d\n",filename,*numObjs);
            printf("File %s numCoords = %d\n",filename,*numCoords);
        }

        /* allocate space for objects[][] and read all objects */
        len = (*numObjs) * (*numCoords);
        objects    = (float**)malloc((*numObjs) * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc(len * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)
            objects[i] = objects[i-1] + (*numCoords);

        i = 0;
        /* read all objects */
        while (fgets(line, lineLen, infile) != NULL) {
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<(*numCoords); j++) {
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
                if (_debug && i == 0) /* print the first object */
                    printf("object[i=%d][j=%d]=%f\n",i,j,objects[i][j]);
            }
            i++;
        }
        assert(i == *numObjs);

        fclose(infile);
        free(line);
    }

    return objects;
}

/*---< read_n_objects() >-----------------------------------------------------*/
int read_n_objects(int     isBinaryFile,  /* flag: 0 or 1 */
                   char   *filename,      /* input file name */
                   int     numObjs,       /* no. objects */
                   int     numCoords,     /* no. coordinates */
                   float **objects)       /* [numObjs][numCoords] */
{
    int i, j, len;

    if (isBinaryFile) {  /* using MPI-IO to read file concurrently */
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: open file %s (err=%s)\n",filename,strerror(errno));
            return 0;
        }
        /* read and discard the first 2 integers, numObjs and numCoords */
        read(infile, &i, sizeof(int));
        read(infile, &i, sizeof(int));

        /* read the objects */
        read(infile, objects[0], numObjs * numCoords * sizeof(float));

        close(infile);
    }
    else {  /* input file is in ASCII format -------------------------------*/
        FILE *infile;
        char *line, *ret;
        int   lineLen;

        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: open file %s (err=%s)\n",filename,strerror(errno));
            return 0;
        }

        /* first find the max length of each line */
        lineLen = MAX_CHAR_PER_LINE;
        line = (char*) malloc(lineLen);
        assert(line != NULL);

        while (fgets(line, lineLen, infile) != NULL) {
            /* check each line to find the max line length */
            while (strlen(line) == lineLen-1) {
                /* this line read is not complete */
                len = strlen(line);
                fseek(infile, -len, SEEK_CUR);

                /* increase lineLen */
                lineLen += MAX_CHAR_PER_LINE;
                line = (char*) realloc(line, lineLen);
                assert(line != NULL);

                ret = fgets(line, lineLen, infile);
                assert(ret != NULL);
            }
        }
        rewind(infile);

        /* read numObjs objects */
        for (i=0; i<numObjs; i++) {
            fgets(line, lineLen, infile);
            if (strtok(line, " \t\n") == NULL) continue;
            for (j=0; j<numCoords; j++)
                objects[i][j] = atof(strtok(NULL, " ,\t\n"));
        }
        fclose(infile);
        free(line);
    }
    return 1;
}

/*---< file_write() >---------------------------------------------------------*/
int file_write(char      *filename,     /* input file name */
               int        numClusters,  /* no. clusters */
               int        numObjs,      /* no. data objects */
               int        numCoords,    /* no. coordinates (local) */
               float    **clusters,     /* [numClusters][numCoords] centers */
               int       *membership,   /* [numObjs] */
               int        verbose)
{
    FILE *fptr;
    int   i, j;
    char  outFileName[1024];

    /* output: the coordinates of the cluster centres ----------------------*/
    sprintf(outFileName, "%s.cluster_centres", filename);
    if (verbose) printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
                        numClusters, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numCoords; j++)
            fprintf(fptr, "%f ", clusters[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    /* output: the closest cluster centre to each of the data points --------*/
    sprintf(outFileName, "%s.membership", filename);
    if (verbose) printf("Writing membership of N=%d data objects to file \"%s\"\n",
                        numObjs, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numObjs; i++)
        fprintf(fptr, "%d %d\n", i, membership[i]);
    fclose(fptr);

    return 1;
}
double wtime(void) 
{
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /* in seconds */
               ((double)etstart.tv_usec) / 1000000.0;  /* in microseconds */
    return now_time;
}
