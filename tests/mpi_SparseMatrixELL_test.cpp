#include <gtest/gtest.h>
#include <yakutat/mpi/SparseMatrix/SparseMatrixELL.hpp>
#include <omp.h>
#include <mpi.h>
using namespace yakutat::backend;

extern int my_argc;
extern char** my_argv;
extern int myid, nproc;


class mpi_SparseMatrixELL_test: public testing::Test
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

TEST_F(mpi_SparseMatrixELL_test, default_constructor)
{
  yakutat::mpi::SparseMatrixELL<double, 4> mat;


  EXPECT_EQ(0, mat.col());
  EXPECT_EQ(0, mat.row());
  EXPECT_EQ(0, mat.max_row_nnz());

}


TEST_F(mpi_SparseMatrixELL_test, Constructor1)
{
  yakutat::mpi::SparseMatrixELL<double, 3> mat(10,10);

  EXPECT_EQ(10, mat.col());
  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(3, mat.max_row_nnz());
}


TEST_F(mpi_SparseMatrixELL_test, Constructor2)
{
  yakutat::mpi::SparseMatrixELL<double, 3> mat(10);

  EXPECT_EQ(10, mat.col());
  EXPECT_EQ(10, mat.row());
  EXPECT_EQ(3, mat.max_row_nnz());
}

TEST_F(mpi_SparseMatrixELL_test, at_and_set)
{
  yakutat::mpi::SparseMatrixELL<double, 3> mat(10);

  mat.set(2, 3, 10.256);
  EXPECT_DOUBLE_EQ(10.256, mat(2,3));
  EXPECT_DOUBLE_EQ(10.256, mat.at(2,3));
  mat.set(2, 3, 1054.2654466);
  EXPECT_DOUBLE_EQ(1054.2654466, mat(2,3));
  EXPECT_DOUBLE_EQ(1054.2654466, mat.at(2,3));
  mat.set(2, 3, 0);
  EXPECT_DOUBLE_EQ(0, mat(2,3));
  EXPECT_DOUBLE_EQ(0, mat.at(2,3));
}


TEST_F(mpi_SparseMatrixELL_test, clear_and_resize)
{
  yakutat::mpi::SparseMatrixELL<double, 3> mat(10);
  mat.set(2, 3, 10.256);
  mat.set(1, 3, 1056);
  mat.set(0, 0, 6);

  mat.clear();

  EXPECT_EQ(0, mat.col());
  EXPECT_EQ(0, mat.row());
  EXPECT_EQ(3, mat.max_row_nnz());
  EXPECT_DOUBLE_EQ(0, mat(2,3));

  mat.clear();

  mat.resize(3,3);
  EXPECT_EQ(3, mat.col());
  EXPECT_EQ(3, mat.row());
  EXPECT_EQ(3, mat.max_row_nnz());
  mat.set(1, 2, 1056);
  EXPECT_DOUBLE_EQ(1056, mat(1,2));

  mat.clear();

  mat.resize(8);
  EXPECT_EQ(8, mat.col());
  EXPECT_EQ(8, mat.row());
  EXPECT_EQ(3, mat.max_row_nnz());
  mat.set(1, 2, 1056);
  EXPECT_DOUBLE_EQ(1056, mat(1,2));
}



TEST_F(mpi_SparseMatrixELL_test, multiply_vector1)
{
  size_t n = 4;
  yakutat::mpi::SparseMatrixELL<double, 4> test_mat(n);
  std::vector<double> rhs(n), x(n);
  mpi_tools world;
  EXPECT_TRUE(world.MPI_division(n, myid, nproc));

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
              test_mat.set(i,j, i+j + 2*i + j);
          }
      }
      test_mat.set(i,i, (i+2)*12);
      rhs[i] = (i+1)^2;
  }

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
              test_mat.set(i,j, test_mat(j,i));
          }
      }
  }
  
  EXPECT_TRUE(test_mat.partition(myid, nproc));

  x = test_mat * rhs;
  world.MPI_vector_collect(x, 1);
  EXPECT_DOUBLE_EQ(132, x[0]);
  EXPECT_DOUBLE_EQ(83, x[1]);
  EXPECT_DOUBLE_EQ(144, x[2]);
  EXPECT_DOUBLE_EQ(400, x[3]);
  
}

TEST_F(mpi_SparseMatrixELL_test, multiply_vector2)
{
  size_t n = 4;
  mpi_tools world;
  EXPECT_TRUE(world.MPI_division(n, myid, nproc));
  yakutat::mpi::SparseMatrixELL<double, 4> test_mat(n);
  std::vector<double> rhs(n), x(n);
  

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
              test_mat.set(i,j, i+j + 2*i + j);
          }
      }
      test_mat.set(i,i, (i+2)*12);
      rhs[i] = (i+1)^2;
  }

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
              test_mat.set(i,j, test_mat(j,i));
          }
      }
  }
  
  EXPECT_TRUE(test_mat.partition(world));

  x = test_mat * rhs;
  world.MPI_vector_collect(x, 1);
  EXPECT_DOUBLE_EQ(132, x[0]);
  EXPECT_DOUBLE_EQ(83, x[1]);
  EXPECT_DOUBLE_EQ(144, x[2]);
  EXPECT_DOUBLE_EQ(400, x[3]);
  
}

TEST_F(mpi_SparseMatrixELL_test, multiply_vector3)
{
  size_t n = 4;
  yakutat::mpi::SparseMatrixELL<double, 4> test_mat(n);
  std::vector<double> rhs(n), x(n);
  mpi_tools world;
  EXPECT_TRUE(world.MPI_division(n, myid, nproc));

  test_mat.set(0,0,1.2562752752);
  test_mat.set(0,3,100.254725725435);
  test_mat.set(3,3,23.22542452572445723);
  test_mat.set(2,0,2354.2272445723);
  test_mat.set(1,2,15.489456341);

  rhs[0] = 456.146541654;
  rhs[1] = 12316.146541654;
  rhs[2] = 5498.146541654;
  rhs[3] = 45665.1454;

  EXPECT_TRUE(test_mat.partition(myid, nproc));
  

  x = test_mat * rhs;
  EXPECT_TRUE(world.MPI_vector_collect(x, 1));
  EXPECT_DOUBLE_EQ(4578719.67291125766, x[0]);
  EXPECT_DOUBLE_EQ(85163.30081336977, x[1]);
  EXPECT_DOUBLE_EQ(1073872.61587928029, x[2]);
  EXPECT_DOUBLE_EQ(1060592.38794393338, x[3]);

}


TEST_F(mpi_SparseMatrixELL_test, multiply_add_subtract_matrix1)
{
  size_t n = 4;
  yakutat::mpi::SparseMatrixELL<double, 4> test_mat(n), copy_mat, ans_mat;
  yakutat::mpi::SparseMatrixELL<double, 4> ans1_mat, ans2_mat;
  mpi_tools world;
  EXPECT_TRUE(world.MPI_division(n, myid, nproc));

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
            test_mat.set(i,j, i+j + 2*i + j);
          }
      }
      test_mat.set(i,i, (i+2)*12);
  }

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
            test_mat.set(i,j, test_mat(j,i));
          }
      }
  }
  test_mat.set(3,3,0.654894658);
  test_mat.set(3,0,1564.215);
  test_mat.set(0,0,0);
  test_mat.set(0,3,0);
  copy_mat = test_mat;
  
  EXPECT_TRUE(test_mat.partition(myid, nproc));
  ans_mat = test_mat * copy_mat;
  EXPECT_TRUE(ans_mat.partition(myid, nproc));
  ans_mat.MPI_allgather(0);
  
  EXPECT_DOUBLE_EQ(45, ans_mat(0,0));
  EXPECT_DOUBLE_EQ(17362.365, ans_mat(1,0));
  EXPECT_DOUBLE_EQ(20646.795, ans_mat(2,0));
  EXPECT_DOUBLE_EQ(1135.39604746347, ans_mat(3,0));

  EXPECT_DOUBLE_EQ(156, ans_mat(0,1));
  EXPECT_DOUBLE_EQ(1490, ans_mat(1,1));
  EXPECT_DOUBLE_EQ(833, ans_mat(2,1));
  EXPECT_DOUBLE_EQ(5199.848841238, ans_mat(3,1));

  EXPECT_DOUBLE_EQ(312, ans_mat(0,2));
  EXPECT_DOUBLE_EQ(833, ans_mat(1,2));
  EXPECT_DOUBLE_EQ(2573, ans_mat(2,2));
  EXPECT_DOUBLE_EQ(10105.803630554, ans_mat(3,2));

  EXPECT_DOUBLE_EQ(111, ans_mat(0,3));
  EXPECT_DOUBLE_EQ(507.203841238, ans_mat(1,3));
  EXPECT_DOUBLE_EQ(720.513630554, ans_mat(2,3));
  EXPECT_DOUBLE_EQ(290.42888701307694, ans_mat(3,3));

  ans1_mat = ans_mat + test_mat; 
  EXPECT_TRUE(ans1_mat.partition(myid, nproc));
  ans1_mat.MPI_allgather(1);
  EXPECT_DOUBLE_EQ(45, ans1_mat(0,0));
  EXPECT_DOUBLE_EQ(17365.365, ans1_mat(1,0));
  EXPECT_DOUBLE_EQ(20652.795, ans1_mat(2,0));
  EXPECT_DOUBLE_EQ(2699.61104746347, ans1_mat(3,0));

  EXPECT_DOUBLE_EQ(159, ans1_mat(0,1));
  EXPECT_DOUBLE_EQ(1526, ans1_mat(1,1));
  EXPECT_DOUBLE_EQ(841, ans1_mat(2,1));
  EXPECT_DOUBLE_EQ(5210.848841238, ans1_mat(3,1));

  EXPECT_DOUBLE_EQ(318, ans1_mat(0,2));
  EXPECT_DOUBLE_EQ(841, ans1_mat(1,2));
  EXPECT_DOUBLE_EQ(2621, ans1_mat(2,2));
  EXPECT_DOUBLE_EQ(10118.803630554, ans1_mat(3,2));

  EXPECT_DOUBLE_EQ(111, ans1_mat(0,3));
  EXPECT_DOUBLE_EQ(518.203841238, ans1_mat(1,3));
  EXPECT_DOUBLE_EQ(733.513630554, ans1_mat(2,3));
  EXPECT_DOUBLE_EQ(291.08378167107694, ans1_mat(3,3));

  ans2_mat = ans_mat - test_mat; 
  EXPECT_TRUE(ans2_mat.partition(myid, nproc));
  ans2_mat.MPI_allgather(1);
  EXPECT_DOUBLE_EQ(45, ans2_mat(0,0));
  EXPECT_DOUBLE_EQ(17359.365, ans2_mat(1,0));
  EXPECT_DOUBLE_EQ(20640.795, ans2_mat(2,0));
  EXPECT_DOUBLE_EQ(-428.81895253653, ans2_mat(3,0));

  EXPECT_DOUBLE_EQ(153, ans2_mat(0,1));
  EXPECT_DOUBLE_EQ(1454, ans2_mat(1,1));
  EXPECT_DOUBLE_EQ(825, ans2_mat(2,1));
  EXPECT_DOUBLE_EQ(5188.848841238, ans2_mat(3,1));

  EXPECT_DOUBLE_EQ(306, ans2_mat(0,2));
  EXPECT_DOUBLE_EQ(825, ans2_mat(1,2));
  EXPECT_DOUBLE_EQ(2525, ans2_mat(2,2));
  EXPECT_DOUBLE_EQ(10092.803630554, ans2_mat(3,2));

  EXPECT_DOUBLE_EQ(111, ans2_mat(0,3));
  EXPECT_DOUBLE_EQ(496.203841238, ans2_mat(1,3));
  EXPECT_DOUBLE_EQ(707.513630554, ans2_mat(2,3));
  EXPECT_DOUBLE_EQ(289.77399235507694, ans2_mat(3,3));
}


TEST_F(mpi_SparseMatrixELL_test, accumulate)
{
  yakutat::mpi::SparseMatrixELL<double, 3> test_mat(5);
  test_mat.set(1,2,3.02);
  test_mat.accumulate(1,2,10.1);
  EXPECT_DOUBLE_EQ(10.1+3.02, test_mat(1,2));
}


TEST_F(mpi_SparseMatrixELL_test, multiply_add_subtract_matrix2)
{
  size_t n = 4;
  mpi_tools world;
  EXPECT_TRUE(world.MPI_division(n, myid, nproc));
  yakutat::mpi::SparseMatrixELL<double, 4> test_mat(n), copy_mat, ans_mat;
  yakutat::mpi::SparseMatrixELL<double, 4> ans1_mat, ans2_mat;
  

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
            test_mat.set(i,j, i+j + 2*i + j);
          }
      }
      test_mat.set(i,i, (i+2)*12);
  }

  for(size_t i=0; i<n; ++i)
  {
      for(size_t j=0; j<n; ++j)
      {
          if(i!=j)
          {
            test_mat.set(i,j, test_mat(j,i));
          }
      }
  }
  test_mat.set(3,3,0.654894658);
  test_mat.set(3,0,1564.215);
  test_mat.set(0,0,0);
  test_mat.set(0,3,0);
  copy_mat = test_mat;
  
  EXPECT_TRUE(test_mat.partition(world));
  ans_mat = test_mat * copy_mat;
  EXPECT_TRUE(ans_mat.partition(world));
  ans_mat.MPI_allgather(0);
  
  EXPECT_DOUBLE_EQ(45, ans_mat(0,0));
  EXPECT_DOUBLE_EQ(17362.365, ans_mat(1,0));
  EXPECT_DOUBLE_EQ(20646.795, ans_mat(2,0));
  EXPECT_DOUBLE_EQ(1135.39604746347, ans_mat(3,0));

  EXPECT_DOUBLE_EQ(156, ans_mat(0,1));
  EXPECT_DOUBLE_EQ(1490, ans_mat(1,1));
  EXPECT_DOUBLE_EQ(833, ans_mat(2,1));
  EXPECT_DOUBLE_EQ(5199.848841238, ans_mat(3,1));

  EXPECT_DOUBLE_EQ(312, ans_mat(0,2));
  EXPECT_DOUBLE_EQ(833, ans_mat(1,2));
  EXPECT_DOUBLE_EQ(2573, ans_mat(2,2));
  EXPECT_DOUBLE_EQ(10105.803630554, ans_mat(3,2));

  EXPECT_DOUBLE_EQ(111, ans_mat(0,3));
  EXPECT_DOUBLE_EQ(507.203841238, ans_mat(1,3));
  EXPECT_DOUBLE_EQ(720.513630554, ans_mat(2,3));
  EXPECT_DOUBLE_EQ(290.42888701307694, ans_mat(3,3));

  ans1_mat = ans_mat + test_mat; 
  EXPECT_TRUE(ans1_mat.partition(world));
  ans1_mat.MPI_allgather(1);
  EXPECT_DOUBLE_EQ(45, ans1_mat(0,0));
  EXPECT_DOUBLE_EQ(17365.365, ans1_mat(1,0));
  EXPECT_DOUBLE_EQ(20652.795, ans1_mat(2,0));
  EXPECT_DOUBLE_EQ(2699.61104746347, ans1_mat(3,0));

  EXPECT_DOUBLE_EQ(159, ans1_mat(0,1));
  EXPECT_DOUBLE_EQ(1526, ans1_mat(1,1));
  EXPECT_DOUBLE_EQ(841, ans1_mat(2,1));
  EXPECT_DOUBLE_EQ(5210.848841238, ans1_mat(3,1));

  EXPECT_DOUBLE_EQ(318, ans1_mat(0,2));
  EXPECT_DOUBLE_EQ(841, ans1_mat(1,2));
  EXPECT_DOUBLE_EQ(2621, ans1_mat(2,2));
  EXPECT_DOUBLE_EQ(10118.803630554, ans1_mat(3,2));

  EXPECT_DOUBLE_EQ(111, ans1_mat(0,3));
  EXPECT_DOUBLE_EQ(518.203841238, ans1_mat(1,3));
  EXPECT_DOUBLE_EQ(733.513630554, ans1_mat(2,3));
  EXPECT_DOUBLE_EQ(291.08378167107694, ans1_mat(3,3));

  ans2_mat = ans_mat - test_mat; 
  EXPECT_TRUE(ans2_mat.partition(world));
  ans2_mat.MPI_allgather(1);
  EXPECT_DOUBLE_EQ(45, ans2_mat(0,0));
  EXPECT_DOUBLE_EQ(17359.365, ans2_mat(1,0));
  EXPECT_DOUBLE_EQ(20640.795, ans2_mat(2,0));
  EXPECT_DOUBLE_EQ(-428.81895253653, ans2_mat(3,0));

  EXPECT_DOUBLE_EQ(153, ans2_mat(0,1));
  EXPECT_DOUBLE_EQ(1454, ans2_mat(1,1));
  EXPECT_DOUBLE_EQ(825, ans2_mat(2,1));
  EXPECT_DOUBLE_EQ(5188.848841238, ans2_mat(3,1));

  EXPECT_DOUBLE_EQ(306, ans2_mat(0,2));
  EXPECT_DOUBLE_EQ(825, ans2_mat(1,2));
  EXPECT_DOUBLE_EQ(2525, ans2_mat(2,2));
  EXPECT_DOUBLE_EQ(10092.803630554, ans2_mat(3,2));

  EXPECT_DOUBLE_EQ(111, ans2_mat(0,3));
  EXPECT_DOUBLE_EQ(496.203841238, ans2_mat(1,3));
  EXPECT_DOUBLE_EQ(707.513630554, ans2_mat(2,3));
  EXPECT_DOUBLE_EQ(289.77399235507694, ans2_mat(3,3));
}

TEST_F(mpi_SparseMatrixELL_test, data)
{
  yakutat::mpi::SparseMatrixELL<double, 3> test_mat(10,10);
  test_mat.set(0,0,5.2);
  test_mat.set(0,3,1.2);
  test_mat.set(1,3,23.2);
  auto data = test_mat.data();
  auto col_idx = std::get<0>(data);
  auto val     = std::get<1>(data);
  EXPECT_DOUBLE_EQ(col_idx[0*3+0], 0);
  EXPECT_DOUBLE_EQ(col_idx[0*3+1], 3);
  EXPECT_DOUBLE_EQ(col_idx[1*3], 3);

  EXPECT_DOUBLE_EQ(val[0*3+0], 5.2);
  EXPECT_DOUBLE_EQ(val[0*3+1], 1.2);
  EXPECT_DOUBLE_EQ(val[1*3+0], 23.2);
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
