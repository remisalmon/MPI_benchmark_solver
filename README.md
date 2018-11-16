# MPI benchmark solver

C/MPI parallelization of the heat equation with source term:

![du/dt - div(grad(u)) = f](http://www.sciweavers.org/upload/Tex2Img_1542331645/eqn.png)

with benchmark functions:

![u(x,y) = cos(a\*x)\*sin(b\*y), f = (a^2+b^2)*u](http://www.sciweavers.org/upload/Tex2Img_1542331701/eqn.png) (steady state equation solved with Jacobi method)

or

![u(x,y,t) = cos(a\*x)\*sin(b\*y)*exp(-c\*t), f = (a^2+b^2-c)*u](http://www.sciweavers.org/upload/Tex2Img_1542331759/eqn.png) (solved with Euler explicit method)

Run:

`mpicc MPI_benchmark_solver.c -o MPI_benchmark_solver && mpirun -np N MPI_benchmark_solver`

with `N = 1+2\*R+2\*C, R = 1,2,..., C = 0,1,2,...`
