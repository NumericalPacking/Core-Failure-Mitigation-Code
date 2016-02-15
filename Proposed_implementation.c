#include "main_funcs.h"

extern "C" void openblas_set_num_threads(int);
using namespace std;
//Proposed implementation. 3 nodes for data
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
	
	
	struct{
		int64_t signextendk: (k-1);
	}s;
	int64_t u_a1b1, u_a1b2_a2b1, u_a2b2, u_a2b1, u_a2b2_a3b1, u_a3b2;
	register int64_t Curr_C1, Curr_C2;
	int location;
	int halfquery_size = query_size / 2;
	
	float* Query_row  = (float*)malloc(data_dim*query_size*sizeof(float));
	int* Query_row_int = (int*)malloc(query_size*data_dim*sizeof(int));
	double** Packed_Data_perthread = (double**)malloc(total_threads * sizeof(double*));
	double** Packed_Result_perthread=(double**)malloc(total_threads * sizeof(double*));
	int** Result_perthread = (int**)malloc(total_threads * sizeof(int*));
	int** top_three_each_node = allocarray_int(query_size,12);
	double* Packed_Query = (double*)malloc(data_dim*halfquery_size * sizeof(double));
	
	for(int i=0; i< total_threads;i++){
		Packed_Data_perthread[i] = (double*)malloc(data_size_per_thread*data_dim*sizeof(double));
		Packed_Result_perthread[i] = (double*)malloc(data_size_per_thread*query_size/2*sizeof(double));
		Result_perthread[i] = (int*)malloc(data_size_per_thread*query_size*sizeof(int));
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
	
	
	

	char* database_path = "****";				//please ensure database is packed with the same k and that all requirements are met as in the paper to avoid oveerlap and overflow
	char* extension= "_cores.bin";
	char str_threads_used[10];
	snprintf(str_threads_used,10,"%d",total_threads);

	
	MPI_Status status;

	/***** Initializations *****/
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
	printf ("MPI task %d has started...\n", taskid);


	/***** Master task only ******/
	if (taskid == MASTER){
		Query_row= ReadFile(Queryfile,query_size);
		copy(Query_row,Query_row+(data_dim*query_size),Query_row_int);
		printf("Proposed with query_size= %d running %d times\n",query_size,time_to_run);
		printf("Actual nodes used: %d\n", numtasks - 1);
		for(int i=0; i<total_threads; i++) {
			char threadname[10];											//modify this section based on your DatabBase naming convention
			snprintf(threadname,10,"%d",i);
			char myfilename[300];
			strcpy(myfilename,database_path);
			strcat(myfilename,threadname);
			strcat(myfilename,str_threads_used);
			strcat(myfilename,extension);
			Packed_Data_perthread[i] = ReadFile_double(myfilename,data_size_per_thread);
		}
		for(int i=1;i<numtasks	;i++)
		{
			printf("sending....\n");
			for (int j=0;j<nthreads_per_node;j++)
			{
				MPI_Send(&Packed_Data_perthread[(i-1)*4 +j][0],data_size_per_thread*data_dim,MPI_DOUBLE,i,i*4+50+j,MPI_COMM_WORLD);
				printf("sent %d to node %d\n",j+1,i);
			}
			
		}
		int** All_top_three = allocarray_int(query_size, 48) ;
		MPI_Datatype mysubarray;
		int starts[2] = {0,0};
		int subsizes[2]  = {query_size,12};
		int bigsizes[2] = {query_size,48};
		MPI_Type_create_subarray(2,bigsizes, subsizes,  starts, MPI_ORDER_C, MPI_INT, &mysubarray);
		MPI_Type_commit(&mysubarray);
		
		for(int ij =0 ; ij<time_to_run;ij++)
		{
			startwtime= MPI_Wtime();
			//Let the master pack the Query
			#pragma omp parallel for private (location)
			for (int x1 = 0; x1 < query_size/2; x1 ++)
			{
				int x = 2 * x1;
				location = x1 *data_dim;
				for (int y = 0; y < data_dim; y ++)
				{
					Packed_Query[location + y] = (Query_row_int[x*data_dim + y] << k) + Query_row_int[(x + 1)*data_dim + y];
				}
			} 
			for(i=1;i<numtasks;i++)
			{
				MPI_Send(&Packed_Query[0],halfquery_size*data_dim,MPI_DOUBLE,i,5,MPI_COMM_WORLD);
			}
			
			//final sorting
			
			
			MPI_Recv(&(All_top_three[0][0]), 1, mysubarray, 1, 11, MPI_COMM_WORLD,&status);
			MPI_Recv(&(All_top_three[0][12]), 1, mysubarray, 1, 12, MPI_COMM_WORLD,&status);
			MPI_Recv(&(All_top_three[0][24]), 1, mysubarray, 3, 13, MPI_COMM_WORLD,&status);
			MPI_Recv(&(All_top_three[0][36]), 1, mysubarray, 3, 14, MPI_COMM_WORLD,&status);
			
			

			//final sorting, we just get the master to do this joor
#pragma omp parallel for
			for(i=0;i<query_size;i++)
			{
				sort(All_top_three[i],All_top_three[i]+48,greaterthanfunction);
			}
			endwtime= MPI_Wtime();
			average_time += (endwtime-startwtime);
			printf("Time to complete: %f\n",endwtime-startwtime);
		}
		printf("\nAverage Time to complete Proposed: %f secs\n\n",(average_time/time_to_run));
		MPI_Type_free(&mysubarray);

	}
	/***** Non-master tasks only *****/

	else if (taskid > MASTER ) {
		//struct timeval start, end;
		for(int i=0; i<nthreads_per_node;i++)
		{   
			MPI_Recv(&Packed_Data_perthread[(taskid-1)*4+i][0],data_size_per_thread*data_dim,MPI_DOUBLE,0,taskid*4+50+i,MPI_COMM_WORLD,&status);
		}
		for(int ij =0 ; ij<time_to_run;ij++)
		{	
			MPI_Recv(&Packed_Query[0],halfquery_size*data_dim,MPI_DOUBLE,0,5,MPI_COMM_WORLD,&status);
			//gettimeofday(&start, NULL);
			#pragma omp parallel
			{
				int id= omp_get_thread_num();
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,halfquery_size,data_size_per_thread, data_dim, 1.0,Packed_Query,data_dim,Packed_Data_perthread[(taskid-1)*4 +id],data_dim,0.0,Packed_Result_perthread[(taskid-1)*4 +id],data_size_per_thread);
			}
			// gettimeofday(&end, NULL);
			//float secs = (end.tv_sec - start.tv_sec);
			//float millis = ((secs*1000000) + end.tv_usec) - (start.tv_usec);
			//cout<<setprecision(6)<<"Time for gemm in taskid= "<<taskid<<" is "<<millis/1000<<" millisecs"<<endl; 	
			
			//we assume node 4 fails
			if(taskid==2){
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,1,6,MPI_COMM_WORLD);
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4+1][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,1,7,MPI_COMM_WORLD);
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4+2][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,1,8,MPI_COMM_WORLD);
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4+3][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,1,9,MPI_COMM_WORLD);
				
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,3,6,MPI_COMM_WORLD);
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4+1][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,3,7,MPI_COMM_WORLD);
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4+2][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,3,8,MPI_COMM_WORLD);
				MPI_Send(&Packed_Result_perthread[(taskid-1)*4+3][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,3,9,MPI_COMM_WORLD);
			}
			
			if(taskid==1)
			{
				MPI_Datatype mysubarray;
				int starts[2] = {0,0};
				int subsizes[2]  = {query_size,12};
				MPI_Type_create_subarray(2, subsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
				MPI_Type_commit(&mysubarray);
				MPI_Recv(&Packed_Result_perthread[4][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,6,MPI_COMM_WORLD,&status);
				MPI_Recv(&Packed_Result_perthread[5][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,7,MPI_COMM_WORLD,&status);
				MPI_Recv(&Packed_Result_perthread[6][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,8,MPI_COMM_WORLD,&status);
				MPI_Recv(&Packed_Result_perthread[7][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,9,MPI_COMM_WORLD,&status);
				//unpack
				#pragma omp parallel firstprivate(s) private (Curr_C2,u_a1b1, u_a1b2_a2b1, u_a2b2, u_a2b1, u_a2b2_a3b1, u_a3b2,Curr_C1)
				{
					for (int id = 0; id < 4; id++){	
#pragma omp for
						for (int i = 0; i < data_size_per_thread*query_size/2; i++)
						{
							int i2 = i * 2;
							Curr_C1 = Packed_Result_perthread[(taskid-1)*4+id][i];
							u_a1b1 = (int64_t(Curr_C1 & 0xFFFFFFFF00000000) >> (k*2));	//r0top
							u_a1b2_a2b1 = (int64_t(Curr_C1 & 0xFFFF0000) >> k);	//r0bot+r4top
							u_a2b2 = (Curr_C1 & 0xFFFF);	//r4bot

							u_a1b2_a2b1 = s.signextendk = u_a1b2_a2b1;

							u_a2b2 = s.signextendk = u_a2b2;
							u_a1b1 += ((u_a1b2_a2b1 >> (k-1)) & 0x1);
							u_a1b2_a2b1 += ((u_a1b1 >> (k-1)) & 0x1);
						
							Curr_C2 = Packed_Result_perthread[taskid*4 + id][i];
							u_a2b1 = (int64_t(Curr_C1 & 0xFFFFFFFF00000000) >> (k*2));	//r0top
							u_a2b2_a3b1 = (int64_t(Curr_C1 & 0xFFFF0000) >> k);	//r0bot+r4top
							//u_a3b2 = (Curr_C1 & 0xFFFF);	//r4bot

							u_a2b2_a3b1 = s.signextendk = u_a2b2_a3b1;
							//u_a3b2 = s.signextendk = u_a3b2;
							u_a2b1 += ((u_a2b2_a3b1 >> (k-1)) & 0x1);
							//u_a2b2_a3b1 += ((u_a2b1 >> (k-1)) & 0x1);
							
							Result_perthread[id][i2] = u_a1b1;
							Result_perthread[id][i2 + 1] = u_a1b2_a2b1 - u_a2b1;
							Result_perthread[4 + id][i2] = u_a2b1;
							Result_perthread[4 + id ][i2 + 1] = u_a2b2;
							
						}
					}
				}
				
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
							value_in_mat = Result_perthread[(taskid-1)*4+id][loc*data_size_per_thread + ind];
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
				
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 11, MPI_COMM_WORLD);
				
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
							value_in_mat = Result_perthread[(taskid)*4+id][loc*data_size_per_thread + ind];
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
				
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 12, MPI_COMM_WORLD);
				MPI_Type_free(&mysubarray);	
			}
			if(taskid==3)
			{
				//processing threads 8-15
				MPI_Datatype mysubarray;
				int starts[2] = {0,0};
				int subsizes[2]  = {query_size,12};
				MPI_Type_create_subarray(2, subsizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &mysubarray);
				MPI_Type_commit(&mysubarray);
				
				MPI_Recv(&Packed_Result_perthread[4][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,6,MPI_COMM_WORLD,&status);
				MPI_Recv(&Packed_Result_perthread[5][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,7,MPI_COMM_WORLD,&status);
				MPI_Recv(&Packed_Result_perthread[6][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,8,MPI_COMM_WORLD,&status);
				MPI_Recv(&Packed_Result_perthread[7][0],halfquery_size*data_size_per_thread,MPI_DOUBLE,2,9,MPI_COMM_WORLD,&status);
				
				//unpack
				#pragma omp parallel firstprivate(s) private (Curr_C2,u_a1b1, u_a1b2_a2b1, u_a2b2, u_a2b1, u_a2b2_a3b1, u_a3b2,Curr_C1)
				{
					for (int id = 0; id < 4; id++){	
#pragma omp for
						for (int i = 0; i < data_size_per_thread*query_size/2; i++)
						{
							
							int i2 = i * 2;
							Curr_C1 = Packed_Result_perthread[(taskid-2)*4+id][i];
							//u_a1b1 = (int64_t(Curr_C1 & 0xFFFFFFFF00000000) >> (k*2));	//r0top
							//u_a1b2_a2b1 = (int64_t(Curr_C1 & 0xFFFF0000) >> k);	//r0bot+r4top
							u_a2b2 = (Curr_C1 & 0xFFFF);	//r4bot

							//u_a1b2_a2b1 = s.signextendk = u_a1b2_a2b1;

							u_a2b2 = s.signextendk = u_a2b2;
							//u_a1b1 += ((u_a1b2_a2b1 >> (k-1)) & 0x1);
							//u_a1b2_a2b1 += ((u_a1b1 >> (k-1)) & 0x1);
													

							Curr_C2 = Packed_Result_perthread[(taskid-1)*4 + id][i];
							u_a2b1 = (int64_t(Curr_C1 & 0xFFFFFFFF00000000) >> (k*2));	//r0top
							u_a2b2_a3b1 = (int64_t(Curr_C1 & 0xFFFF0000) >> k);	//r0bot+r4top
							u_a3b2 = (Curr_C1 & 0xFFFF);	//r4bot

							u_a2b2_a3b1 = s.signextendk = u_a2b2_a3b1;

							u_a3b2 = s.signextendk = u_a3b2;
							u_a2b1 += ((u_a2b2_a3b1 >> (k-1)) & 0x1);
							u_a2b2_a3b1 += ((u_a2b1 >> (k-1)) & 0x1);


							Result_perthread[id +8][i2] = u_a2b1;
							Result_perthread[id+8][i2 + 1] = u_a2b2;
							Result_perthread[12 + id][i2] = u_a2b2_a3b1-u_a2b2;
							Result_perthread[12 + id ][i2 + 1] = u_a3b2;

						}
					}
				}
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
							value_in_mat = Result_perthread[(taskid-1)*4+id][loc*data_size_per_thread + ind];
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
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 13, MPI_COMM_WORLD);
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
							value_in_mat = Result_perthread[(taskid)*4+id][loc*data_size_per_thread + ind];
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
				MPI_Send(&(top_three_each_node[0][0]), 1, mysubarray, 0, 14, MPI_COMM_WORLD);
				MPI_Type_free(&mysubarray);	
			}
			
			
		}
	}
	
	MPI_Finalize();
	
	for(int i=0; i< total_threads;i++){
		free(Packed_Data_perthread[i]);
		free(Packed_Result_perthread[i]);
		free(Result_perthread[i]);
	}

	
	free(Query_row);
	free(Query_row_int);
	free(Packed_Data_perthread);
	free(Packed_Result_perthread);
	free(Result_perthread);
	free(top_three_each_node);
	free(Packed_Query);
	return 0;
}



