# MPI benchmark solver

MPI parallelization of iterative solvers for the heat equation with benchmark function:

![du/dt - div(grad(u)) = f](http://www.sciweavers.org/upload/Tex2Img_1542331645/eqn.png)

with

![u(x,y) = cos(a\*x)\*sin(b\*y), f = (a^2+b^2)*u](http://www.sciweavers.org/upload/Tex2Img_1542331701/eqn.png) (steady state Poisson equation solved with Jacobi method)

or

![u(x,y,t) = cos(a\*x)\*sin(b\*y)*exp(-c\*t), f = (a^2+b^2-c)*u](http://www.sciweavers.org/upload/Tex2Img_1542331759/eqn.png) (heat equation solved with Euler explicit method)

Run:

`mpicc mpi_benchmark_solver.c -o mpi_exec && mpirun -np N ./mpi_exec`

with N = 1+2\*R+2\*C, R = 1,2,..., C = 0,1,2,...
