 #!/bin/bash

#echo export OMP_NUM_THREADS=4
#echo export OPENBLAS_NUM_THREADS=1
rm gemm
mpic++ -o gemm $3 -w -fopenmp -lblas -m64
ssh node001 ls /home/MPI_Vlad_Test/MPI_Vlad_Test > rubbish.txt
ssh node002 ls /home/MPI_Vlad_Test/MPI_Vlad_Test > rubbish.txt
ssh node003 ls /home/MPI_Vlad_Test/MPI_Vlad_Test > rubbish.txt
ssh node004 ls /home/MPI_Vlad_Test/MPI_Vlad_Test > rubbish.txt
mpirun -np $1 ./gemm $2 $4 $5
