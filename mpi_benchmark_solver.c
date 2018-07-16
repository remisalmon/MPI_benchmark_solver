#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZEX 100
#define SIZEY 100
#define ITERATIONS 1000
#define MASTER 0
#define START 1
#define END 2
#define COMM 3

#define DT 0.001
#define LX 2*3.1416
#define LY 2*3.1416
#define DX LX/SIZEX
#define DY LY/SIZEY

void initialization(int x, int y, double* data); //initial conditions
void update(int x, int y, int i, int j, int t, double* A, double* B, int id, int* dims); //update local matrix B
void upgrade(int x, int y, int t, double* A, double* B, int id, int* dims); //copy matrix B to inner matrix of A
void save(int x, int y, double* data, int n); //save to format data_mpi[n].txt
double function_f(int x, int y, int t); //function f of "du/dt - Delta(u) = f"
double benchmark_u(int x, int y, int t); //function u (benchmark)

int main(int argc, char *argv[])
{
	int id, nbTasks, nbSlaves, sizexLocal, sizeyLocal, source, dest, north, south, east, west, message, rc, i, j;
	double tstart, tend;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Request request[8];
	MPI_Datatype colA;
	MPI_Datatype colB;
	MPI_Datatype rowA;
	MPI_Datatype rowB;
	MPI_Datatype matrixA;
	MPI_Datatype matrixB;
	
	nbSlaves = nbTasks-1; //1 master + nbTasks-1 slaves
	
	int dims[2]; //parallel topology
	dims[0] = 0;
	dims[1] = 0;
	MPI_Dims_create(nbSlaves, 2, dims); //calculate topology with dims[0] = dimX, dims[1] = dimY
	
	sizexLocal = SIZEX/dims[0]; //local domain size in X
	sizeyLocal = SIZEY/dims[1]; //local domain size in Y
	
	MPI_Type_vector(sizexLocal+2, 1, sizeyLocal+2, MPI_DOUBLE, &colA); //column datatype for matrix A
	MPI_Type_commit(&colA);
	MPI_Type_vector(sizexLocal, 1, sizeyLocal, MPI_DOUBLE, &colB); //column datatype for matrix B
	MPI_Type_commit(&colB);
	
	MPI_Type_vector(sizeyLocal+2, 1, 1, MPI_DOUBLE, &rowA); //row datatype for matrix A
	MPI_Type_commit(&rowA);
	MPI_Type_vector(sizeyLocal, 1, 1, MPI_DOUBLE, &rowB); //row datatype for matrix B
	MPI_Type_commit(&rowB);
	
	MPI_Type_vector(sizexLocal+2, sizeyLocal+2, SIZEY+2, MPI_DOUBLE, &matrixA); //datatype for matrix A
	MPI_Type_commit(&matrixA);
	MPI_Type_vector(sizexLocal, sizeyLocal, SIZEY, MPI_DOUBLE, &matrixB); //datatype for matrix B
	MPI_Type_commit(&matrixB);
	
	if(id == MASTER)
	{
		if(nbSlaves < 1 || SIZEX%dims[0] != 0 || SIZEY%dims[1] != 0)
		{
			MPI_Abort(MPI_COMM_WORLD, rc);
			exit(1);
		}
		
		printf("MPI init with %d slaves, topology %dx%d\n", nbSlaves, dims[0], dims[1]);
		
		tstart = MPI_Wtime();
		
		double* data;
		data = malloc((SIZEX+2)*(SIZEY+2)*sizeof(double)); //extra ghost rows and columns, data(i,j)=data[i*WIDTH+j]
		
		initialization(SIZEX+2, SIZEY+2, data);
		
		message = START;
		for(i=1; i<=dims[0]; i++)
		{
			for(j=1; j<=dims[1]; j++)
			{
				dest = (i-1)*dims[1]+j;
				MPI_Send(&data[(i-1)*sizexLocal*(SIZEY+2)+(j-1)*sizeyLocal], 1, matrixA, dest, message, MPI_COMM_WORLD); //send to neighbors, with ghosts cells
			}
		}
		
		free(data);
		data = malloc(SIZEX*SIZEY*sizeof(double));
		
		message = END;
		for(i=1; i<=dims[0]; i++)
		{
			for(j=1; j<=dims[1]; j++)
			{
				source = (i-1)*dims[1]+j;
				MPI_Recv(&data[(i-1)*sizexLocal*SIZEY+(j-1)*(sizeyLocal)], 1, matrixB, source, message, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //receive from neighbors
			}
		}
		
		tend = MPI_Wtime();
		printf("elapsed time = %.02f sec\n", tend-tstart);
		
		//for(i=0; i<SIZEX; i++)
		//{
		//	for(j=0; j<SIZEY; j++)
		//		data[i*SIZEY+j] = pow(data[i*SIZEY+j]-benchmark_u(i+1, j+1, ITERATIONS), 2); //overwrite with L2 norm between result and benchmark function u
		//}
		
		save(SIZEX, SIZEY, data, 0);
		
		free(data);
		
		MPI_Finalize();
	}
	
	if(id != MASTER)
	{	
		int t;
		double* A; //matrix A with ghost cells
		double* B; //matrix B for local computation
		
		A = malloc((sizexLocal+2)*(sizeyLocal+2)*sizeof(double));
		B = malloc(sizexLocal*sizeyLocal*sizeof(double));
		
		source = MASTER;
		message = START;
		MPI_Recv(&A[0], (sizexLocal+2)*(sizeyLocal+2), MPI_DOUBLE, source, message, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //reception des lignes dans A
		
		message = COMM;
		west = id-1;
		east = id+1;
		north = id-dims[1];
		south = id+dims[1];
		if(west < 1 || west > nbSlaves || id%dims[1] == 1 || dims[1] == 1)
			west = MPI_PROC_NULL;
		if(east < 1 || east > nbSlaves || id%dims[1] == 0)
			east = MPI_PROC_NULL;
		if(north < 1 || north > nbSlaves)
			north = MPI_PROC_NULL;
		if(south < 1 || south > nbSlaves)
			south = MPI_PROC_NULL;
		
		for(t=1; t<=ITERATIONS; t++)
		{
			if(id == 1 && (t%(ITERATIONS/10)) == 0)
				printf("iteration %d\n", t);
			
			for(j=1; j<sizeyLocal+2-1; j++) //update sides of B
			{
				i = 1;
				update(sizexLocal+2, sizeyLocal+2, i, j, t, A, B, id, dims);
				i = sizexLocal+2-1-1;
				update(sizexLocal+2, sizeyLocal+2, i, j, t, A, B, id, dims);
			}
			for(i=1; i<sizexLocal+2-1; i++) //update sides of B
			{
				j = 1;
				update(sizexLocal+2, sizeyLocal+2, i, j, t, A, B, id, dims);
				j = sizeyLocal+2-1-1;
				update(sizexLocal+2, sizeyLocal+2, i, j, t, A, B, id, dims);
			}
			
			MPI_Isend(&B[0], 1, rowB, north, message, MPI_COMM_WORLD, &request[0]); //send first row of updated B to ghost cell of A
			MPI_Isend(&B[(sizexLocal-1)*sizeyLocal], 1, rowB, south, message, MPI_COMM_WORLD, &request[1]); //send new last row of updated B to ghost cell of A
			MPI_Isend(&B[0], 1, colB, west, message, MPI_COMM_WORLD, &request[2]); //send new first column of updated B to ghost cell of A
			MPI_Isend(&B[sizeyLocal-1], 1, colB, east, message, MPI_COMM_WORLD, &request[3]); //send last column of updated B to ghost cell of A
			
			MPI_Irecv(&A[1], 1, rowA, north, message, MPI_COMM_WORLD, &request[4]); //receive data in ghost cell
			MPI_Irecv(&A[(sizexLocal+2-1)*(sizeyLocal+2)+1], 1, rowA, south, message, MPI_COMM_WORLD, &request[5]); //receive data in ghost cell	
			MPI_Irecv(&A[1*(sizeyLocal+2)], 1, colA, west, message, MPI_COMM_WORLD, &request[6]); //receive data in ghost cell
			MPI_Irecv(&A[1*(sizeyLocal+2)+sizeyLocal+2-1], 1, colA, east, message, MPI_COMM_WORLD, &request[7]); //receive data in ghost cell
			
			for(i=1+1; i<sizexLocal+2-1-1; i++) //update inside of B
			{
				for(j=1+1; j<sizeyLocal+2-1-1; j++)
					update(sizexLocal+2, sizeyLocal+2, i, j, t, A, B, id, dims);
			}
			
			upgrade(sizexLocal+2, sizeyLocal+2, t, A, B, id, dims); //update A
			
			MPI_Waitall(8, request, MPI_STATUSES_IGNORE);
		}
		
		dest = MASTER;
		message = END;
		MPI_Send(&B[0], sizexLocal*sizeyLocal, MPI_DOUBLE, dest, message, MPI_COMM_WORLD); //send result to master
		
		free(A);
		free(B);
		
		MPI_Finalize();
	}
	
	return(0);
}

void initialization(int x, int y, double* data)
{
	int i, j;
	for(i=0; i<x; i++)
	{
		for(j=0; j<y; j++)
			//data[i*y+j] = 0; //poisson equation
			data[i*y+j] = benchmark_u(i, j, 0); //heat equation
	}
}

void update(int x, int y, int i, int j, int t, double* A, double* B, int id, int* dims) //(i,j) parcourent A
{	
	//jacobi iteration:
	//B[(i-1)*(y-2)+(j-1)] = (A[(i-1)*y+j] + A[(i+1)*y+j] + A[i*y+(j-1)] + A[i*y+(j+1)] + (DX*DX)*function_f(i+floor((id-1)/dims[1])*(x-2), j+((id-1)%dims[1])*(y-2), 0))/4.0; //DX=DY
	//Euler explicit iteration:
	B[(i-1)*(y-2)+(j-1)] = A[i*y+j] + DT*((A[(i-1)*y+j]-2.0*A[i*y+j]+A[(i+1)*y+j])/(DX*DX) + (A[i*y+j-1]-2.0*A[i*y+j]+A[i*y+j+1])/(DY*DY) + function_f(i+floor((id-1)/dims[1])*(x-2), j+((id-1)%dims[1])*(y-2), t));
}

void upgrade(int x, int y, int t, double* A, double* B, int id, int* dims)
{
	int i, j;
	for(i=1; i<x-1; i++) //update matrix A from matrix B
	{
		for(j=1; j<y-1; j++)
			A[i*y+j] = B[(i-1)*(y-2)+(j-1)];
	}
	
	if(floor((id-1)/dims[1]) == 0) //Dirichlet boundary condition on edges
	{
		for(j=0; j<y; j++)
			A[0*y+j] = benchmark_u(0, j+((id-1)%dims[1])*(y-2), t);
	}
	if(floor((id-1)/dims[1]) == dims[0]-1) //Dirichlet boundary condition on edges
	{
		for(j=0; j<y; j++)
			A[(x-1)*y+j] = benchmark_u(SIZEX+1, j+((id-1)%dims[1])*(y-2), t);
	}
	if(id%dims[1] == 1 || dims[1] == 1) //Dirichlet boundary condition on edges
	{
		for(i=0; i<x; i++)
			A[i*y+0] = benchmark_u(i+floor((id-1)/dims[1])*(x-2), 0, t);
	}
	if(id%dims[1] == 0) //Dirichlet boundary condition on edges
	{
		for(i=0; i<x; i++)
			A[i*y+(y-1)] = benchmark_u(i+floor((id-1)/dims[1])*(x-2), SIZEY+1, t);
	}
}

void save(int x, int y, double* data, int n)
{
	FILE* f;
	char name[13];
	
	sprintf(name, "data_mpi%01d.txt", n);
	f=fopen(name, "w");
	
	int i, j;
	for(i=0; i<x; i++)
	{
		for(j=0; j<y; j++)
			fprintf(f, "%f ", data[i*y+j]);
		fprintf(f, "\n");
	}
	
	fclose(f);
}

double function_f(int x, int y, int t) //function f of "du/dt - Delta(u) = f"
{
	double val;
	//val = -2.0*benchmark_u(x, y, 0); //poisson equation
	val = benchmark_u(x, y, t); //heat equation
	
	return(val);
}

double benchmark_u(int x, int y, int t) //function u (benchmark)
{
	double val;
	//val = sin(x*DX)*cos(y*DY); //poisson equation
	val = sin(x*DX)*cos(y*DY)*exp(-t*DT); //heat equation
	
	return(val);
}
