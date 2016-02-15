#include "main_funcs.h"
extern "C" void openblas_set_num_threads(int);
using namespace std;
//Checksum implementation. 3 nodes for data
int main (int argc, char *argv[])
{
	if(argc<4)
	{
		printf("ERROR, argc is %d\n",argc);
		return 0;
	}
	time_to_run = atoi(argv[1]);
	query_size=atoi(argv[2]);
	data_size=atoi(argv[3]);
	
	//set the number of nodes and number of threads per node on the main_funcs.h accordingly
	omp_set_num_threads(nnodes);
	openblas_set_num_threads(1);
	
	if((data_size%total_threads!=0)  || (data_size%total_threads_chksum!=0))
	{
		printf("Wrong dbsize. Choose dbsize divisible by %d and %d\n",total_threads,total_threads_chksum);
		return 0;
	}
	data_size_per_thread_chksum  = data_size/total_threads_chksum;
	
	float* Query_row  = (float*)malloc(data_dim*query_size*sizeof(float));
	float** Data_perthread = (float**)malloc(total_threads*sizeof(float*));
	float** Result_perthread = (float**)malloc(total_threads*sizeof(float*));
	float** top_three_each_node = allocarray(query_size,12);

	startwtime=0.0;
	average_time=0;
	
	for(int i=0; i< total_threads;i++){			//16 total threads
		Data_perthread[i] = (float*)malloc(data_size_per_thread_chksum*data_dim*sizeof(float));
		Result_perthread[i] = (float*)malloc(data_size_per_thread_chksum*query_size*sizeof(float));
	}
	
	__m128 load_data,load_data1,add_data;
	/*if(data_size_per_thread_chksum>9344)
	{
		printf("Please choose a smaller dbsize\n");
		//because 9344 is the max we have;
		return 0;
	}*/
	
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
	startwtime=0.0;
	average_time=0;
		
	char *ext;
	char* database_path = "";										//absolute path_to_database bin file eg. ./Image/ChecksumDB82944.bin
	char str_threads_used[10];										//modify this based on your DB naming convention
	snprintf(str_threads_used,10,"%d",total_threads);
	ext= ".bin";
	
	//Ensure Database for checksum node contains checksum data


	MPI_Status status;

	/***** Initializations *****/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	printf ("MPI task %d has started...\n", taskid);

	
	/***** Master task only ******/
	if (taskid == MASTER){		//we use the xxx12.bin files as they contain the checksum
		printf("Checksum with query_size= %d running %d times\n",query_size,time_to_run);
		printf("Actual nodes used: %d\n",numtasks-1);
		Query_row= ReadFile(Queryfile,query_size);
		for(int i=0; i<total_threads; i++) {
			char threadname[10];								//modify this section based on your DB naming convention
			snprintf(threadname,10,"%d",i);
			char myfilename[300];
			strcpy(myfilename,database_path);
			strcat(myfilename,threadname);
			strcat(myfilename,str_threads_used);
			strcat(myfilename,extension);
			Data_perthread[i] = ReadFile(myfilename,data_size_per_thread_chksum);
		}
		
		for(int i=1;i<numtasks;i++)
		{
		//here
			printf("sending....\n");
			for (int j=0;j<nthreads_per_node;j++)
			{
				MPI_Send(&Data_perthread[(i-1)*nthreads_per_node][0],data_size_per_thread_chksum*data_dim,MPI_FLOAT,i,j+1,MPI_COMM_WORLD);
				printf("sent %d to node %d\n",j+1,i);
			}
		}
		float** All_top_three = allocarray(query_size, 3*total_threads_chksum) ;
			MPI_Datatype mysubarray;
			int starts[2] = {0,0};
			int subsizes[2]  = {query_size,3*nnodes};
			int bigsizes[2] = {query_size,3*total_threads_chksum};
			MPI_Type_create_subarray(2,bigsizes, subsizes,  starts, MPI_ORDER_C, MPI_INT, &mysubarray);
			MPI_Type_commit(&mysubarray);
		for(int ij=0;ij<time_to_run;ij++)
		{
			startwtime= MPI_Wtime();
			for(int i=1;i<numtasks;i++)
			{
				MPI_Send(&Query_row[0],query_size*data_dim,MPI_FLOAT,i,nthreads_per_node+1,MPI_COMM_WORLD);
			}
			//receive from all but  taskid = 1
			//final sorting
			for(int i=2;i<(numtasks-1);i++)
			{
				MPI_Recv(&(All_top_three[0][(i-1)*3*nnodes]), 1, mysubarray, i, nthreads_per_node+21, MPI_COMM_WORLD,&status);
			} 
			MPI_Recv(&(All_top_three[0][0]), 1, mysubarray, (numtasks-1), nthreads_per_node+21, MPI_COMM_WORLD,&status);
		
			//final sorting, we just get the master to do this joor
			#pragma omp parallel for
			for(int i=0;i<query_size;i++)
			{
				sort(All_top_three[i],All_top_three[i]+3*total_threads_chksum,greaterthanfunction);
			}
			endwtime= MPI_Wtime();
			average_time += (endwtime-startwtime);
			printf("Time to complete: %f\n",endwtime-startwtime);
		}
		printf("\nAverage Time to complete Checksum: %f secs\n\n",(average_time/time_to_run));
		MPI_Type_free(&mysubarray);
		

	}  /* end of master section */
	/***** Non-master tasks only *****/

	else if (taskid > MASTER ) {
		
		for(int i=0; i<nthreads_per_node;i++)
		{
			MPI_Recv(&Data_perthread[(taskid-1)*nthreads_per_node][0],data_size_per_thread_chksum*data_dim,MPI_FLOAT,0,i+1,MPI_COMM_WORLD,&status);
		}
		for(int ij=0;ij<time_to_run;ij++)
		{
			MPI_Recv(&Query_row[0],query_size*data_dim,MPI_FLOAT,0,nthreads_per_node+1,MPI_COMM_WORLD,&status);

			#pragma omp parallel
			{
				int id= omp_get_thread_num();
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,query_size,data_size_per_thread_chksum, data_dim, 1.0,Query_row,data_dim,Data_perthread[(taskid-1)*nthreads_per_node +id],data_dim,0.0,Result_perthread[(taskid-1)*nthreads_per_node +id],data_size_per_thread_chksum);
			}

		//we assume taskid 1 failed	
		//this experiment is for 4 nodes only. Please design the Checksum implentation to suport more than this.
			if(taskid==2)
			{
				//This node has data for threads 4 and 5 already as he computed them
				MPI_Recv(&Result_perthread[8][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,3,10,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[12][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,11,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[9][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,3,13,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[13][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,14,MPI_COMM_WORLD,&status);
				MPI_Send(&Result_perthread[6][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,3,15,MPI_COMM_WORLD);
				MPI_Send(&Result_perthread[7][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,18,MPI_COMM_WORLD);
				
				//recover 
				#pragma omp parallel private(load_data,load_data1,add_data)
				{
					#pragma omp for
					for(int i=0;i<query_size*data_size_per_thread_chksum/4;i++)
					{
						int j = i*4;
						load_data = _mm_load_ps(Result_perthread[4]+j);
						load_data1 = _mm_load_ps(Result_perthread[8]+j);
						add_data = _mm_load_ps(Result_perthread[12]+j);
						add_data=(_mm_sub_ps(add_data, _mm_add_ps(load_data,load_data1)));
						_mm_store_ps(Result_perthread[0]+j, add_data);
					}

					#pragma omp for
					for(int i=0;i<query_size*data_size_per_thread_chksum/4;i++)
					{
						int j = i*4;
						load_data = _mm_load_ps(Result_perthread[5]+j);
						load_data1 = _mm_load_ps(Result_perthread[9]+j);
						add_data = _mm_load_ps(Result_perthread[13]+j);
						add_data=(_mm_sub_ps(add_data, _mm_add_ps(load_data,load_data1)));
						_mm_store_ps(Result_perthread[1]+j, add_data);
					}

				}
				//Send the recovered to  4 to sort
				MPI_Send(&Result_perthread[0][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,21,MPI_COMM_WORLD);
				MPI_Send(&Result_perthread[1][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,22,MPI_COMM_WORLD);
				#pragma omp parallel  private (values_to_retain,value_in_mat)
				{
					int id = omp_get_thread_num();
					for (int loc=0;loc<query_size;loc++)
					{
						values_to_retain.clear();
						values_to_retain.resize(data_size_per_thread_chksum);
						int counts=0;
						for(int ind=0;ind<data_size_per_thread_chksum;ind++)
						{
							value_in_mat = Result_perthread[(taskid-1)*4+id][loc*data_size_per_thread_chksum + ind];
							if(value_in_mat>threshold)
							{
								values_to_retain[counts]= value_in_mat;
								counts++;
							}
						}
						sort(values_to_retain.begin(),values_to_retain.end(),greaterthanfunction);
						//Here we are putting 3 top images per core in a 
						top_three_each_node[loc][id*3] =values_to_retain[0];
						top_three_each_node[loc][id*3 +1] = values_to_retain[1];
						top_three_each_node[loc][id*3+2] = values_to_retain[2];
					}
				}
				MPI_Datatype mysubarray;
				int starts[2] = {0,0};
				int subsizes[2]  = {query_size,12};
				MPI_Type_create_subarray(2, subsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
				MPI_Type_commit(&mysubarray);
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 25, MPI_COMM_WORLD);
				MPI_Type_free(&mysubarray);
			}

			if(taskid==3)
			{
				MPI_Send(&Result_perthread[8][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,10,MPI_COMM_WORLD);
				MPI_Send(&Result_perthread[9][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,13,MPI_COMM_WORLD);
				MPI_Recv(&Result_perthread[6][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,15,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[14][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,17,MPI_COMM_WORLD,&status);
				MPI_Send(&Result_perthread[11][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,19,MPI_COMM_WORLD);
				
				#pragma omp parallel private(load_data,load_data1,add_data)
				{
					#pragma omp for
					for(int i=0;i<query_size*data_size_per_thread_chksum/4;i++)
					{
						int j = i*4;
						load_data = _mm_load_ps(Result_perthread[6]+j);
						load_data1 = _mm_load_ps(Result_perthread[10]+j);
						add_data = _mm_load_ps(Result_perthread[14]+j);
						add_data=(_mm_sub_ps(add_data, _mm_add_ps(load_data,load_data1)));
						_mm_store_ps(Result_perthread[2]+j, add_data);
					}

				}
				MPI_Send(&Result_perthread[2][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,4,23,MPI_COMM_WORLD);
				#pragma omp parallel  private (values_to_retain,value_in_mat)
				{
					int id = omp_get_thread_num();
					for (int loc=0;loc<query_size;loc++)
					{
						values_to_retain.clear();
						values_to_retain.resize(data_size_per_thread_chksum);
						int counts=0;
						for(int ind=0;ind<data_size_per_thread_chksum;ind++)
						{
							value_in_mat = Result_perthread[(taskid-1)*4+id][loc*data_size_per_thread_chksum + ind];
							if(value_in_mat>threshold)
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
				int subsizes[2]  = {query_size,12};
				MPI_Type_create_subarray(2, subsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
				MPI_Type_commit(&mysubarray);
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 25, MPI_COMM_WORLD);
				MPI_Type_free(&mysubarray);
				
			}
			if(taskid==4)
			{
				MPI_Send(&Result_perthread[12][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,11,MPI_COMM_WORLD);
				MPI_Send(&Result_perthread[13][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,14,MPI_COMM_WORLD);
				MPI_Send(&Result_perthread[14][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,3,17,MPI_COMM_WORLD);
				MPI_Recv(&Result_perthread[7][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,18,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[11][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,3,19,MPI_COMM_WORLD,&status);
				#pragma omp parallel private(load_data,load_data1,add_data,i,j)
				{
					#pragma omp for
					for(int i=0;i<query_size*data_size_per_thread_chksum/4;i++)
					{
						int j = i*4;
						load_data = _mm_load_ps(Result_perthread[7]+j);
						load_data1 = _mm_load_ps(Result_perthread[11]+j);
						add_data = _mm_load_ps(Result_perthread[15]+j);
						add_data=(_mm_sub_ps(add_data, _mm_add_ps(load_data,load_data1)));
						_mm_store_ps(Result_perthread[3]+j, add_data);
					}
				}
				MPI_Recv(&Result_perthread[2][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,3,23,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[0][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,21,MPI_COMM_WORLD,&status);
				MPI_Recv(&Result_perthread[1][0],query_size*data_size_per_thread_chksum,MPI_FLOAT,2,22,MPI_COMM_WORLD,&status);
				
				#pragma omp parallel  private (values_to_retain,value_in_mat)
				{
					int id = omp_get_thread_num();
					for (int loc=0;loc<query_size;loc++)
					{
						values_to_retain.clear();
						values_to_retain.resize(data_size_per_thread_chksum);
						int counts=0;
						for(int ind=0;ind<data_size_per_thread_chksum;ind++)
						{
							value_in_mat = Result_perthread[id][loc*data_size_per_thread_chksum + ind];
							if(value_in_mat>threshold)
							{
								values_to_retain[counts]= value_in_mat;
								counts++;
							}
						}
						sort(values_to_retain.begin(),values_to_retain.end(),greaterthanfunction);
						//Here we are putting 3 top images per core in a 
						top_three_each_node[loc][id*3] =values_to_retain[0];
						top_three_each_node[loc][id*3 +1] = values_to_retain[1];
						top_three_each_node[loc][id*3+2] = values_to_retain[2];
					}
				}
				MPI_Datatype mysubarray;
				int starts[2] = {0,0};
				int subsizes[2]  = {query_size,12};
				MPI_Type_create_subarray(2, subsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
				MPI_Type_commit(&mysubarray);
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 25, MPI_COMM_WORLD);
				MPI_Type_free(&mysubarray);
			}
		}
	}
	MPI_Finalize();
	return 0;
}



