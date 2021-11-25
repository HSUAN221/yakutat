//  mpicxx -std=c++11 -o3 -o debug_hsuan debug_hsuan.cpp
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>
#include <mpi.h>

#include "stopWatch.hpp"
#include "os.hpp"
#include <yakutat/backend/mpi_tools.hpp>
#include <yakutat/backend/Matrix_operations.hpp>

#include <yakutat/mpi/SparseMatrix/SparseMatrixELL.hpp>
#include <yakutat/mpi/SparseMatrix/SparseMatrixCSR.hpp>
#include <yakutat/mpi/DynamicMatrix.hpp>
#include <yakutat/mpi/Algorithms/LinearSolvers/bicgstab.hpp>
#include <yakutat/mpi/Algorithms/LinearSolvers/cg.hpp>


#include <yakutat/SparseMatrix/SparseMatrixELL.hpp>
#include <yakutat/SparseMatrix/SparseMatrixCSR.hpp>
#include <yakutat/DynamicMatrix.hpp>
#include <yakutat/mpi/Algorithms/LinearSolvers/bicgstab.hpp>
#include <yakutat/mpi/Algorithms/LinearSolvers/cg.hpp>


#include <yakutat/Algorithms/Cholesky/LLT.hpp>
#include <yakutat/Algorithms/LU/LD.hpp>




int main(int argc, char **argv)
{   
    size_t n{4};

    // MPI variables
    // ======================================================== //
    int myid, nproc;
    MPI_Init (&argc, &argv);					
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);		
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);	
    yakutat::backend::mpi_tools world;
    world.MPI_division(n, myid, nproc);
    // ======================================================== //

    // OMP variables
    // ======================================================== //
    size_t num_threads;
    num_threads = 2;
    omp_set_num_threads(num_threads);
    // ======================================================== //


    yakutat::mpi::SparseMatrixCSR<double> lhs_mat(n);
    std::vector<double> rhs(n), x(n,0.0);
    lhs_mat.partition(world);

    // ======================================================== //
    for(size_t i=0; i<n; ++i)
    {
        for(size_t j=0; j<n; ++j)
        {
            if(i!=j)
            {
                lhs_mat.set(i,j, i+j + 2*i + j);
            }
        }
        lhs_mat.set(i,i, (i+2)*12);
        rhs[i] = (i+2)*12;
    }

    for(size_t i=0; i<n; ++i)
    {
        for(size_t j=0; j<n; ++j)
        {
            if(i!=j)
            {
                lhs_mat.set(i,j, lhs_mat(j,i));
            }
        }
    }
    auto copy_mat = lhs_mat;
    lhs_mat.set(0,0,0);

    // ======================================================== //

    // std::cout << lhs_mat << std::endl;
    // std::for_each(rhs.begin(), rhs.end(), os<double>);

    auto mat  = lhs_mat * copy_mat;
    mat.partition(world);
    mat.MPI_allgather(1);


    if(myid == 1)
        std::cout << mat << std::endl;

    if(myid == 0)
        std::cout << mat << std::endl;


    MPI_Finalize();    
    return 0;
}