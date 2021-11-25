#include <gtest/gtest.h>
#include <yakutat/matrix_definition.hpp>
#include <yakutat/Algorithms/LinearSolvers/cg.hpp>
#include <yakutat/../debug/debug_tools/stopWatch.hpp>
#include <omp.h>
#include <mpi.h>
using namespace yakutat::backend;



class cg_test: public testing::Test
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

      }

      virtual void SetUp() override {}


      virtual void TearDown() override {}



      static void TearDownTestCase()
      {

      }
};


TEST_F(cg_test, cg_SparseMatrixELL)
{
    stopWatch timer;
    size_t n = 4;
    yakutat::SparseMatrixELL<double, 4> lhs_mat(n);
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
    yakutat::cg<yakutat::SparseMatrixELL<double, 4>> solver;
    EXPECT_TRUE(solver.setTolerance(1e-20));
    EXPECT_TRUE(solver.initialize(lhs_mat));
    std::tie(iters, error) = solver.solve(rhs, x);
    timer.stop();

    EXPECT_DOUBLE_EQ(0.096738164741718492, x[0]);
    EXPECT_DOUBLE_EQ(-0.034387761958445010, x[1]);
    EXPECT_DOUBLE_EQ(-0.011036037717536564, x[2]);
    EXPECT_DOUBLE_EQ(0.094184839819923400, x[3]);
    std::cout << "cg elapsedTime : " << timer.elapsedTime() << std::endl;
}



TEST_F(cg_test, cg_DynamicMatrix)
{
    stopWatch timer;
    size_t n = 4;
    yakutat::DynamicMatrix<double> lhs_mat(n);
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
    yakutat::cg<yakutat::DynamicMatrix<double>> solver;
    EXPECT_TRUE(solver.setTolerance(1e-20));
    EXPECT_TRUE(solver.initialize(lhs_mat));
    std::tie(iters, error) = solver.solve(rhs, x);
    timer.stop();
    

    EXPECT_NEAR(0.096738164741718492, x[0], 1e-10);
    EXPECT_NEAR(-0.034387761958445010, x[1], 1e-10);
    EXPECT_NEAR(-0.011036037717536564, x[2], 1e-10);
    EXPECT_NEAR(0.094184839819923400, x[3], 1e-10);
    std::cout << "cg elapsedTime : " << timer.elapsedTime() << std::endl;
}

int main(int argc, char **argv) 
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}