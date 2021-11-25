#include <gtest/gtest.h>
#include <yakutat/mpi/matrix_definition.hpp>
#include <yakutat/mpi/Algorithms/LinearSolvers/bicgstab.hpp>
#include <yakutat/../debug/debug_tools/stopWatch.hpp>
#include <omp.h>
#include <mpi.h>
using namespace yakutat::backend;

extern int my_argc;
extern char** my_argv;
extern int myid, nproc;

class mpi_bicgstab_test: public testing::Test
{
    protected:
    

      // For classes that are expensive to set up and the tests do not
      // change the resource, there's no harm in their sharing a single
      // resource copy. In your test class, defines 
      // static void SetUpTestCase() & static void TearDownTestCase().
      // google test will calls the former in the first TEST_P and the
      // latter after the last TEST_P.
      static void SetUpTestCase()
      {
        // MPI variables
        // ======================================================== //
        MPI_Init (&my_argc, &my_argv);					
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);		
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);	
        // ======================================================== //

        // OMP variables
        // ======================================================== //
        size_t num_threads;
        num_threads = 2;
        omp_set_num_threads(num_threads);
        // ======================================================== //
      }

      virtual void SetUp() override {}


      virtual void TearDown() override {}



      static void TearDownTestCase()
      {
        MPI_Finalize(); 
      }
};

TEST_F(mpi_bicgstab_test, bicgstab1)
{
    stopWatch timer;
    size_t n = 4;
    
    yakutat::mpi::SparseMatrixELL<double, 4> lhs_mat(n);
    std::vector<double> rhs(n, 0.0);
    std::vector<double> x(n, 0.0);
    size_t iters;
    double error;

    // Define lhs_mat, rhs_vector
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
        rhs[i] = (i+1)^2;
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
    // ======================================================== //
    timer.start();
    yakutat::mpi::bicgstab<yakutat::mpi::SparseMatrixELL<double, 4>> solver;
    EXPECT_TRUE(solver.initialize(lhs_mat));
    EXPECT_TRUE(solver.partition(myid, nproc));
    EXPECT_TRUE(solver.setTolerance(1e-20));
    std::tie(iters, error) = solver.solve(rhs, x);
    EXPECT_TRUE(solver.MPI_allgather(x, 1));
    timer.stop();

    EXPECT_DOUBLE_EQ(0.096738164741718492, x[0]);
    EXPECT_DOUBLE_EQ(-0.034387761958445010, x[1]);
    EXPECT_DOUBLE_EQ(-0.011036037717536564, x[2]);
    EXPECT_DOUBLE_EQ(0.094184839819923400, x[3]);
    if(myid == 0)
        std::cout << "yakutat::mpi::bicgstab elapsedTime : " << timer.elapsedTime() << std::endl;
}


TEST_F(mpi_bicgstab_test, bicgstab2)
{
    stopWatch timer;
    size_t n = 4;
    mpi_tools world;
    world.MPI_division(n, myid, nproc);
    
    yakutat::mpi::SparseMatrixELL<double, 4> lhs_mat(n);
    std::vector<double> rhs(n, 0.0);
    std::vector<double> x(n, 0.0);
    size_t iters;
    double error;

    // Define lhs_mat, rhs_vector
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
        rhs[i] = (i+1)^2;
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
    // ======================================================== //
    timer.start();
    yakutat::mpi::bicgstab<yakutat::mpi::SparseMatrixELL<double, 4>> solver;
    EXPECT_TRUE(solver.initialize(lhs_mat));
    EXPECT_TRUE(solver.partition(world));
    EXPECT_TRUE(solver.setTolerance(1e-20));
    std::tie(iters, error) = solver.solve(rhs, x);
    EXPECT_TRUE(solver.MPI_allgather(x, 1));
    timer.stop();

    EXPECT_DOUBLE_EQ(0.096738164741718492, x[0]);
    EXPECT_DOUBLE_EQ(-0.034387761958445010, x[1]);
    EXPECT_DOUBLE_EQ(-0.011036037717536564, x[2]);
    EXPECT_DOUBLE_EQ(0.094184839819923400, x[3]);
    if(myid == 0)
        std::cout << "yakutat::mpi::bicgstab elapsedTime : " << timer.elapsedTime() << std::endl;
}


TEST_F(mpi_bicgstab_test, bicgstab_DynamicMatrix)
{
    stopWatch timer;
    size_t n = 4;
    mpi_tools world;
    world.MPI_division(n, myid, nproc);
    yakutat::mpi::DynamicMatrix<double> lhs_mat(n);
    std::vector<double> rhs(n, 0.0);
    std::vector<double> x(n, 0.0);
    size_t iters;
    double error;

    // Define lhs_mat, rhs_vector
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
        rhs[i] = (i+1)^2;
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
    // ======================================================== //
    timer.start();
    yakutat::mpi::bicgstab<yakutat::mpi::DynamicMatrix<double>> solver;
    EXPECT_TRUE(solver.initialize(lhs_mat));
    EXPECT_TRUE(solver.partition(world));
    EXPECT_TRUE(solver.setTolerance(1e-20));
    std::tie(iters, error) = solver.solve(rhs, x);
    EXPECT_TRUE(solver.MPI_allgather(x, 1));
    timer.stop();

    EXPECT_DOUBLE_EQ(0.096738164741718492, x[0]);
    EXPECT_DOUBLE_EQ(-0.034387761958445010, x[1]);
    EXPECT_DOUBLE_EQ(-0.011036037717536564, x[2]);
    EXPECT_DOUBLE_EQ(0.094184839819923400, x[3]);
    if(myid == 0)
        std::cout << "yakutat::mpi::bicgstab elapsedTime : " << timer.elapsedTime() << std::endl;
}

int my_argc;
char** my_argv;
int myid, nproc;
int main(int argc, char **argv) 
{

  testing::InitGoogleTest(&argc, argv);
  my_argc = argc;
  my_argv = argv;
  
  return RUN_ALL_TESTS();
}
