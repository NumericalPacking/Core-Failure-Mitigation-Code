#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <mpi.h>
#include <cblas.h>
//#include <mkl.h>
#include <omp.h>
#include <time.h>
#include <xmmintrin.h>
#include <iostream>
#include <vector>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>
#include <iomanip>
#define  MASTER 0

//change this database especially for checksum to have the correct values

const int data_dim = 8192;
const int k = 16;
int nnodes /*= ****/;						//set this to number nodes on the cluster
int nthreads_per_node /*= ****/;			//set this to number of threads on each node
int total_threads = nnodes*nthreads_per_node;
int total_threads_chksum = (nnodes-1)*nthreads_per_node;
int time_to_run=1;
int query_size,data_size,data_size_per_thread,data_size_per_thread_chksum, numtasks, taskid ;
double startwtime, endwtime, average_time;
vector<float> values_to_retain;
float  value_in_mat;
float threshold=500;


float* ReadFile(char *name, int data_size_per_thread)
{
	FILE *file;
	float *buffer;
	unsigned long fileLen;
	long numFloat;

	//Open file
	file = fopen(name, "rb");
	if (!file)
	{
		fprintf(stderr, "Unable to open file %s", name);
		return NULL;
	}

	//Get file length
	fseek(file, 0, SEEK_END);
	fileLen = ftell(file);
	fseek(file, 0, SEEK_SET);

	numFloat = fileLen / sizeof(float);
	//this helps us choose dbsizes using datadim oof 8192;
	
	numFloat= data_size_per_thread*data_dim;
	printf("Number of float is %d\n",numFloat);
	//Allocate memory
	buffer = (float*)malloc(numFloat*sizeof(float));
	//buffer = new float[numFloat];

	if (!buffer)
	{
		fprintf(stderr, "Memory error!"); fclose(file);
		return NULL;
	}

	//Read file contents into buffer
	{
		unsigned long len_in;
		len_in = fread(buffer, sizeof(float), numFloat, file);
		if (len_in != numFloat)
		{
			//cout << endl << "File " << name << " contained only " << len_in << " elements, while we should be reading " << fileLen << endl;
			printf("\nFile %s contained only %ld elements, while we should be reading %ld\n", name, len_in, fileLen);
			//exit(-1);
		}
	}
	fclose(file);
	
	float * bufferInd = buffer;
	return buffer;
}

bool greaterthanfunction(float i, float j)
{
	return (i > j);
}

float **allocarray(int rw, int col) {
	float *data = (float*) malloc(rw*col*sizeof(float));
	float **arr = (float**) malloc(rw*sizeof(float *));
	for (int i=0; i<rw; i++)
	arr[i] = &(data[i*col]);

	return arr;
}

int **allocarray_int(int rw, int col) {
	int *data = (int*) malloc(rw*col*sizeof(int));
	int **arr = (int**) malloc(rw*sizeof(int *));
	for (int i=0; i<rw; i++)
	arr[i] = &(data[i*col]);

	return arr;
}

double* ReadFile_double(char *name,int data_size_per_thread)
{
	FILE *file;
	double *buffer;
	unsigned long fileLen;
	int numFloat;

	//Open file
	file = fopen(name, "rb");
	if (!file)
	{
		fprintf(stderr, "Unable to open file %s", name);
		return NULL;
	}

	//Get file length
	fseek(file, 0, SEEK_END);
	fileLen = ftell(file);
	fseek(file, 0, SEEK_SET);

	numFloat = fileLen / sizeof(double);
	//this helps us choose dbsizes using datadim oof 8192;
	
	numFloat= data_size_per_thread*data_dim;
	//printf("Number of float is %d\n",numFloat);
	//Allocate memory
	buffer = new double[numFloat];

	if (!buffer)
	{
		fprintf(stderr, "Memory error!"); fclose(file);
		return NULL;
	}

	//Read file contents into buffer
	{
		unsigned long len_in;
		len_in = fread(buffer, sizeof(double), numFloat, file);
		if (len_in != numFloat)
		{
			printf("\nFile %s contained only %ld elements, while we should be reading %ld\n", name, len_in, fileLen);
			exit(-1);
		}
	}
	fclose(file);

	double * bufferInd = buffer;
	return buffer;
}
