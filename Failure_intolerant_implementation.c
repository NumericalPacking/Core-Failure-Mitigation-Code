#include "main_funcs.h"

extern "C" void openblas_set_num_threads(int);
using namespace std;
//Conventional implementation. 
int main (int argc, char *argv[])
{
	if(argc<4)
	{
		printf("ERROR, argc is %d\n",argc);
		return 0;
	}
	
	//set the number of nodes and number of threads per node on the main_funcs.h accordingly
	omp_set_num_threads(nnodes);				
	openblas_set_num_threads(1);
	

	startwtime=0.0;
	average_time=0;
	time_to_run = atoi(argv[1]);
	query_size=atoi(argv[2]);
	data_size=atoi(argv[3]);
	/*if(query_size!=64 &&query_size!=128 &&query_size!=256 &&query_size!=512 &&query_size!=1024 &&query_size!=2048 && query_size!=4096 &&query_size!=8192)
	{
		printf("Wrong query size chosen\n");
		return 0;
	}*/
	if((data_size%total_threads!=0))
	{
		printf("Wrong dbsize. Choose dbsize divisible by %d\n",total_threads);		//this ensures all threads operate on same datasize
		return 0;
	}
	data_size_per_thread  = data_size/total_threads;
	
	/*if(data_size_per_thread>7008)
	{
		printf("Please choose a smaller dbsize\n");
		//because 7008 is the max we have;
		return 0;
	}*/
	
	float* Query_row  = (float*)malloc(data_dim*query_size*sizeof(float));
	float** Data_perthread = (float**)malloc(total_threads*sizeof(float*));
	float** Result_perthread = (float**)malloc(total_threads*sizeof(float*));
	float** top_three_each_node = allocarray(query_size,3*nnodes);
	for(int i=0; i< total_threads;i++){
		Data_perthread[i] = (float*)malloc(data_size_per_thread*data_dim*sizeof(float));
		Result_perthread[i] = (float*)malloc(data_size_per_thread*query_size*sizeof(float));
	}
	
	
	char  Queryfile[300],*pathtofile;  							//change this section to suit naming convention of Queryfile.
	pathtofile = "***";											//absolute path_to_query bin file eg. ./Image/Queryfile288.bin
	strcpy(Queryfile,pathtofile);
	/*
	char str_query_size[10];
	
	snprintf(str_query_size,10,"%d",query_size);
	strcat(Queryfile,str_query_size);
	strcat(Queryfile,ext);*/

	char *ext;
	char* database_path = "";										//absolute path_to_database bin file eg. ./Image/ConventionalDB82944.bin
	char str_threads_used[10];										//modify this based on your DB naming convention
	snprintf(str_threads_used,10,"%d",total_threads);
	ext= ".bin";
	
	
	MPI_Status status;
	/***** Initializations *****/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	printf ("MPI task %d has started...\n", taskid);

	/***** Master task only ******/
	if (taskid == MASTER){
		Query_row= ReadFile(Queryfile, query_size);
		printf("Conventional with query_size= %d running %d times\n",query_size,time_to_run);
		printf("Actual nodes used for computation: %d\n", numtasks - 1 );
		for(int i=0; i<total_threads; i++) {
			char threadname[10];									//modify this section based on your DataBase naming convention
			snprintf(threadname,10,"%d",i);
			char myfilename[300];
			strcpy(myfilename,database_path);
			strcat(myfilename,threadname);
			strcat(myfilename,str_threads_used);
			strcat(myfilename,ext);
			Data_perthread[i] = ReadFile(myfilename,data_size_per_thread);
		}
		//by default we assume these reside at the nodes. Thus, we do not include the cost of performing this
		for(int i=1;i<numtasks;i++)
		{
			printf("sending....\n");
			for (int j=0;j<nthreads_per_node;j++)
			{
				MPI_Send(&Data_perthread[(i-1)*nthreads_per_node +j][0],data_size_per_thread*data_dim,MPI_FLOAT,i,j+1,MPI_COMM_WORLD);
				printf("sent %d to node %d\n",j+1,i);
			}
		}
	float** All_top_three = allocarray(query_size, 3*total_threads) ;	//3 top results from all total_threads threads
			MPI_Datatype mysubarray;
			int starts[2] = {0,0};
			int subsizes[2]  = {query_size,3*nnodes};
			int bigsizes[2] = {query_size,3*total_threads};	
			MPI_Type_create_subarray(2,bigsizes, subsizes,  starts, MPI_ORDER_C, MPI_INT, &mysubarray);
			MPI_Type_commit(&mysubarray);
for(int ij=0;ij<time_to_run;ij++)
		{
			startwtime= MPI_Wtime();
			for(int i=1;i<numtasks;i++)
			{
				MPI_Send(&Query_row[0],query_size*data_dim,MPI_FLOAT,i,nthreads_per_node+1,MPI_COMM_WORLD);
			}
			//we need to sort
			
			for(int i=1;i<numtasks;i++)
			{
				MPI_Recv(&(All_top_three[0][(i-1)*3*nnodes]), 1, mysubarray, i, nthreads_per_node+2, MPI_COMM_WORLD,&status);
			}
			//final sorting, the master does the final sorting.
			#pragma omp parallel for
			for(int i=0;i<query_size;i++)
			{
				sort(All_top_three[i],All_top_three[i]+3*total_threads,greaterthanfunction);
			}
			
			endwtime= MPI_Wtime();
			printf("Time to complete: %f\n",endwtime-startwtime);
			average_time+= (endwtime-startwtime);
		}
		printf("\nAverage Time to complete conventional: %f\n\n",(average_time/time_to_run));
		MPI_Type_free(&mysubarray);
	}  /* end of master section */
	/***** Non-master tasks only *****/

	else if (taskid > MASTER ) {
		
		//struct timeval start, end;
		for(int i=0; i<nthreads_per_node;i++)
		{
			MPI_Recv(&Data_perthread[(taskid-1)*nthreads_per_node +i][0],data_size_per_thread*data_dim,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
		}
		for(int ij=0;ij<time_to_run;ij++){
			MPI_Recv(&Query_row[0],query_size*data_dim,MPI_FLOAT,0,nthreads_per_node+1,MPI_COMM_WORLD,&status);
//gettimeofday(&start, NULL);
#pragma omp parallel 
			{
				int id= omp_get_thread_num();
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,query_size,data_size_per_thread, data_dim, 1.0,Query_row,data_dim,Data_perthread[(taskid-1)*nthreads_per_node +id],data_dim,0.0,Result_perthread[(taskid-1)*nthreads_per_node +id],data_size_per_thread);
			}
//gettimeofday(&end, NULL);
//float secs = (end.tv_sec - start.tv_sec);
//float millis = ((secs*1000000) + end.tv_usec) - (start.tv_usec);
//cout<<setprecision(6)<<"Time for gemm in taskid= "<<taskid<<" is "<<millis/1000<<" secs"<<endl;
			
			//then sort before sending back
#pragma omp parallel  private (values_to_retain,value_in_mat)
			{
				int id = omp_get_thread_num();
				for (int loc=0;loc<query_size;loc++)
				{
					values_to_retain.clear();
					values_to_retain.resize(data_size_per_thread);
					int counts=0;
					for(int ind=0;ind<data_size_per_thread;ind++)
					{
						value_in_mat = Result_perthread[(taskid-1)*nthreads_per_node+id][loc*data_size_per_thread + ind];
						if(value_in_mat>threshold)				//set the threshold depending on your data. a pretest is required for appropriate VLAD match threshold
						{
							values_to_retain[counts]= value_in_mat;
							counts++;
						}
					}
					sort(values_to_retain.begin(),values_to_retain.end(),greaterthanfunction);
					top_three_each_node[loc][id*3] =values_to_retain[0];
					top_three_each_node[loc][id*3 +1] = values_to_retain[1];
					top_three_each_node[loc][id*3+2] = values_to_retain[2];
				}
			}
			MPI_Datatype mysubarray;
			int starts[2] = {0,0};
			int subsizes[2]  = {query_size,3*nthreads_per_node};
			MPI_Type_create_subarray(2, subsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
			MPI_Type_commit(&mysubarray);
			MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, nthreads_per_node+2, MPI_COMM_WORLD);
			MPI_Type_free(&mysubarray);
		}
	}
	MPI_Finalize();
	
	for(int i=0; i< total_threads;i++){
		free(Data_perthread[i]);
		free(Result_perthread[i]);
	}
	
	free(Query_row);
	free(Data_perthread);
	free(Result_perthread);
	free(top_three_each_node);
	

	return 0;
}



