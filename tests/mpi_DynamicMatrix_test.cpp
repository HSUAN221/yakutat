#include <gtest/gtest.h>
#include <yakutat/mpi/DynamicMatrix.hpp>
#include <yakutat/backend/Matrix_traits.hpp>
#include <omp.h>
#include <mpi.h>
using namespace yakutat::mpi;
using namespace yakutat::backend;

extern int my_argc;
extern char** my_argv;
extern int myid, nproc;



class mpi_DynamicMatrix_test: public testing::Test
{
    protected:
      using MatrixType = DynamicMatrix<double>;
      using size_type = typename MatrixTraits<MatrixType>::size_type;
      using value_type = typename MatrixTraits<MatrixType>::value_type;
      using array_size_type = typename MatrixTraits<MatrixType>::array_size_type;
      using array_value_type = typename MatrixTraits<MatrixType>::array_value_type;


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
        size_type num_threads;
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


TEST_F(mpi_DynamicMatrix_test, default_constructor)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat;


  EXPECT_EQ(0, mat.col());
  EXPECT_EQ(0, mat.row());

}


TEST_F(mpi_DynamicMatrix_test, constructor)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat(10, 6), mat1(20);
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat2(10, 6,10.34);
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat3(mat2);


  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(6, mat.col());

  EXPECT_EQ(20, mat1.row());
  EXPECT_EQ(20, mat1.col());

  for(size_type i=0; i<mat2.row(); ++i)
  {
    for(size_type j=0; j<mat2.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(10.34, mat2(i, j));
      EXPECT_DOUBLE_EQ(10.34, mat3(i, j));
    }
  }

}


TEST_F(mpi_DynamicMatrix_test, set)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat(10, 6);


  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(6, mat.col());


  for(size_type i=0; i<mat.row(); ++i)
  {
    for(size_type j=0; j<mat.col(); ++j)
    {
      mat.set(i, j, 10.34);
      EXPECT_DOUBLE_EQ(10.34, mat(i, j));
    }
  }

}


TEST_F(mpi_DynamicMatrix_test, accumulate)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat(10, 6);


  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(6, mat.col());


  for(size_type i=0; i<mat.row(); ++i)
  {
    for(size_type j=0; j<mat.col(); ++j)
    {
      mat.set(i, j, 10.34);
      mat.accumulate(i, j, 100.2);
      EXPECT_DOUBLE_EQ(10.34+100.2, mat(i, j));
    }
  }
}



TEST_F(mpi_DynamicMatrix_test, clear_and_empty)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat(10, 6);


  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(6, mat.col());


  for(size_type i=0; i<mat.row(); ++i)
  {
    for(size_type j=0; j<mat.col(); ++j)
    {
      mat.set(i, j, 10.34);
      mat.accumulate(i, j, 100.2);
    }
  }
  mat.clear();
  EXPECT_EQ(0, mat.row());
  EXPECT_EQ(0, mat.col());

  for(size_type i=0; i<mat.row(); ++i)
  {
    for(size_type j=0; j<mat.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(0, mat(i, j));
    }
  }

  EXPECT_TRUE(mat.empty());

}


TEST_F(mpi_DynamicMatrix_test, resize_and_clear)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat;

  mat.resize(10, 6, 10.34);
  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(6, mat.col());


  for(size_type i=0; i<mat.row(); ++i)
  {
    for(size_type j=0; j<mat.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(10.34, mat(i, j));
    }
  }

  mat.clear();
  mat.resize(12, 10);
  EXPECT_EQ(12, mat.row());
  EXPECT_EQ(10, mat.col());

  mat.clear();
  mat.resize(15);
  EXPECT_EQ(15, mat.row());
  EXPECT_EQ(15, mat.col());


}


TEST_F(mpi_DynamicMatrix_test, data_and_fill)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat;

  mat.resize(2, 3, 10.34);
  EXPECT_EQ(2, mat.row());
  EXPECT_EQ(3, mat.col());


  for(size_type i=0; i<mat.row(); ++i)
  {
    for(size_type j=0; j<mat.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(10.34, mat(i, j));
    }
  }

  auto data = mat.data();
  for(auto iter=data.begin(); iter!=data.end(); ++iter)
  {
    EXPECT_DOUBLE_EQ(10.34, *iter);
  }

  mat.fill(100.2);

  for(auto iter=mat.data().begin(); iter!=mat.data().end(); ++iter)
  {
    EXPECT_DOUBLE_EQ(100.2, *iter);
  }

}




TEST_F(mpi_DynamicMatrix_test, mpi_add_subract1)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat_A, mat_B, mat_C;

  mat_A.resize(10,10,100.2231);
  mat_B.resize(10,10,120.8);
  mat_A.partition(myid, nproc);
  mat_C = mat_A + mat_B;
  mat_C.partition(myid, nproc);
  mat_C.MPI_allgather(2);


  for(size_type i=0; i<mat_C.row(); ++i)
  {
    for(size_type j=0; j<mat_C.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(100.2231+120.8, mat_C(i, j));
    }
  }

  mat_C = mat_A - mat_B;
  mat_C.partition(myid, nproc);
  mat_C.MPI_allgather(3);
  for(size_type i=0; i<mat_C.row(); ++i)
  {
    for(size_type j=0; j<mat_C.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(100.2231-120.8, mat_C(i, j));
    }
  }

}

TEST_F(mpi_DynamicMatrix_test, mpi_add_subract2)
{
  DynamicMatrix<mpi_DynamicMatrix_test::value_type> mat_A, mat_B, mat_C;
  mpi_tools world;


  mat_A.resize(10,10,100.2231);
  mat_B.resize(10,10,120.8);
  world.MPI_division(mat_A.row(), myid, nproc);

  mat_A.partition(world);
  mat_C = mat_A + mat_B;
  mat_C.partition(world);
  mat_C.MPI_allgather(2);


  for(size_type i=0; i<mat_C.row(); ++i)
  {
    for(size_type j=0; j<mat_C.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(100.2231+120.8, mat_C(i, j));
    }
  }

  mat_C = mat_A - mat_B;
  mat_C.partition(world);
  mat_C.MPI_allgather(3);
  for(size_type i=0; i<mat_C.row(); ++i)
  {
    for(size_type j=0; j<mat_C.col(); ++j)
    {
      EXPECT_DOUBLE_EQ(100.2231-120.8, mat_C(i, j));
    }
  }
}

TEST_F(mpi_DynamicMatrix_test, mpi_multiply_vector)
{

}


TEST_F(mpi_DynamicMatrix_test, mpi_multiply_matrix)
{
  
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
