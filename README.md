# Core-Failure-Mitigation-In-Integer-Sum-of-Product-Computations-On-Cloud-Computing-Systems

We have attached in this folder the source code for the Amazon AWS image matchin experiment of section VI of our submitted manuscript


The folder contains the source code for *Failure_intolerant_implementation.c
					*Checksum_implementation.c 
					*Checksum_implementation_nofailure.c and 
					*Proposed_implementation.c.
	It also contains a sample vlad-processed image database with seven hundred and seventeen 8192-length VLAD image signatures (717-by-8192 matrix) 
										
We assume a 5 node Linux parallel computing cluster with four threads per node.
One node is set up as the master while the others are setup as slaves.
It is easy to generalize the code for up to N-nodes by changing the variables (shown as comments) in the source codes.
If Amazon spot instances are to be used, please follow the setup as outlined in http://star.mit.edu/cluster 
Any database of images can be used for the tests provided they are compacted using the VLAD algorithm. (cf references [58][59] of the paper)

Each code can be compiled using the MPI command:

mpic++ -o gemm [filename] -w -fopenmp -lblas -m64

The compiled code can be execution by specifying the following paramenters:

mpirun -np [num_nodes] ./gemm [timestorun] [query_size] [dbsize]

where:  [filename] - any of the sourcecodes
	    [timestorun] - times to run the code for an average execution times
		[query_size] - size of query: 0<query_size<dbsize. 
		[dbsize]	- size of the image database 
		
Note: Please ensure realtime update of the executable "gemm" on all nodes to ensure the same version of code is executed on all nodes.
