# MPI benchmark solver

MPI parallelization of iterative solvers for the heat equation with benchmark function:

du/dt - div(grad(u)) = f

with

u(x,y) = cos(a\*x)\*sin(b\*y) and f = (a^2+b^2)*u (steady state Poisson equation solved with jacobi method)

or

u(x,y,t) = cos(a\*x)\*sin(b\*y)*exp(-c\*t) and f = (a^2+b^2-c)*u (heat equation solved with Euler explicit method)

Run:

\> mpicc mpi_benchmark_solver.c -o mpi_exec && mpirun -np N ./mpi_exec

with N = 1+2\*R+2\*C, R = 1,2,..., C = 0,1,2,...
