#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../../src/cpp/core/include/pch.hpp"
#include <iostream>
#include <cmath>
#include <iterator>
#include <algorithm>

class OperationTest : public ::testing::Test
{
protected:
	OperationManager *cpuopmanager = new OperationManager(OperationManager::device_types::CPU_DEVICE);
	OperationManager *gpuopmanager = new OperationManager(OperationManager::device_types::GPU_DEVICE);

	const float relative_tolerance = 1e-6;
	const float absolute_tolerance = 1e-6;

	float *matrix1;
	float *matrix2;
	float *matrix3;
	float *matrix4;
	float *result_matrix;
	int rows1 = 3;
	int cols1 = 3;
	int rows2 = 4;
	int cols2 = 4;

	void SetUp() override
	{
		matrix1 = new float[rows1 * cols1];
		matrix2 = new float[rows1 * cols1];
		matrix3 = new float[rows2 * cols2];
		matrix4 = new float[rows2 * cols2];

		matrix1[0] = 1.0;matrix1[1] = 2.0;matrix1[2] = 3.0;
		matrix1[3] = 4.0;matrix1[4] = 5.0;matrix1[5] = 6.0;
		matrix1[6] = 7.0;matrix1[7] = 8.0;matrix1[8] = 9.0;

		matrix2[0] = 2.0;matrix2[1] = 0.0;matrix2[2] = 1.0;
		matrix2[3] = 1.0;matrix2[4] = 2.0;matrix2[5] = 3.0;
		matrix2[6] = 4.0;matrix2[7] = 1.0;matrix2[8] = 2.0;


		matrix3[0] = 5;matrix3[1] = 9.7;matrix3[2] = 1;matrix3[3] = 6.2;
		matrix3[4] = 12;matrix3[5] = 91;matrix3[6] = 15;matrix3[7] = 4.7;
		matrix3[8] = 19;matrix3[9] = 74;matrix3[10] = 3.2;matrix3[11] = 9.1;
		matrix3[12] = 3.1;matrix3[13] = 82;matrix3[14] = 31;matrix3[15] = 22;

		matrix4[0] = 7.5; matrix4[1] = 2.3; matrix4[2] = 8.1; matrix4[3] = 4.6;
		matrix4[4] = 11.2; matrix4[5] = 5.9; matrix4[6] = 14.3; matrix4[7] = 3.8;
		matrix4[8] = 17.6; matrix4[9] = 6.4; matrix4[10] = 2.1; matrix4[11] = 8.9;
		matrix4[12] = 1.5; matrix4[13] = 9.2; matrix4[14] = 12.7; matrix4[15] = 4.3;
	}

	void TearDown() override
	{
		delete cpuopmanager;
		delete gpuopmanager;
		delete[] matrix1;
		delete[] matrix2;
		delete[] matrix3;
		delete[] matrix4;
	}
};

auto check_result = [](float actual, float expected, float rel_tol, float abs_tol) {
    if (fabs(expected) < abs_tol) {
        // Use absolute error for values very close to zero
        return fabs(actual - expected) < abs_tol;
    } else {
        // Use relative error for other values
        return fabs((actual - expected) / expected) < rel_tol;
    }
};

TEST_F(OperationTest, Determinant_Test)
{
	result_matrix = cpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix1, rows1, cols1);
	EXPECT_TRUE(check_result(result_matrix[0], 0, relative_tolerance, absolute_tolerance))
		<< "CPU det(matrix1) = " << result_matrix[0] << ", expected 0";
	delete[] result_matrix;

	result_matrix = cpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix2, rows1, cols1);
	EXPECT_TRUE(check_result(result_matrix[0], -5, relative_tolerance, absolute_tolerance))
		<< "CPU det(matrix2) = " << result_matrix[0] << ", expected -5";
	delete[] result_matrix;

	result_matrix = gpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix1, rows1, cols1);
	EXPECT_TRUE(check_result(result_matrix[0], 0, relative_tolerance, absolute_tolerance))
		<< "GPU det(matrix1) = " << result_matrix[0] << ", expected 0";
	delete[] result_matrix;

	result_matrix = gpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix2, rows1, cols1);
	EXPECT_TRUE(check_result(result_matrix[0], -5, relative_tolerance, absolute_tolerance))
		<< "GPU det(matrix2) = " << result_matrix[0] << ", expected -5";
	delete[] result_matrix;

	result_matrix = cpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix3, rows2, cols2);
	EXPECT_TRUE(check_result(result_matrix[0], -26398.6062, relative_tolerance, absolute_tolerance))
		<< "CPU det(matrix3) = " << result_matrix[0] << ", expected -26398.6062";
	delete[] result_matrix;

	result_matrix = cpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix4, rows2, cols2);
	EXPECT_TRUE(check_result(result_matrix[0], -4655.8174, relative_tolerance, absolute_tolerance))
		<< "CPU det(matrix4) = " << result_matrix[0] << ", expected -4655.8174";
	delete[] result_matrix;

	result_matrix = gpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix3, rows2, cols2);
	EXPECT_TRUE(check_result(result_matrix[0], -26398.6062, relative_tolerance, absolute_tolerance))
		<< "GPU det(matrix3) = " << result_matrix[0] << ", expected -26398.6062";
	delete[] result_matrix;

	result_matrix = gpuopmanager->single_vector_op(operation_types::DETERMINANT, matrix4, rows2, cols2);
	EXPECT_TRUE(check_result(result_matrix[0], -4655.8174, relative_tolerance, absolute_tolerance))
		<< "GPU det(matrix4) = " << result_matrix[0] << ", expected -4655.8174";
	delete[] result_matrix;
}
TEST_F(OperationTest, Element_Wise_Add_Test)
{

}

TEST_F(OperationTest, Element_Wise_Sub_Test)
{
}

TEST_F(OperationTest, Element_Wise_Div_Test)
{
}

TEST_F(OperationTest, Element_Wise_Mul_Test)
{
}

TEST_F(OperationTest, Frobenius_Norm_Test)
{
}

TEST_F(OperationTest, Matrix_Multiplication_Test)
{
}

TEST_F(OperationTest, Trace_Test)
{
}

TEST_F(OperationTest, Transpose_Test)
{
    size_t array1_size = 9;

    float testarray1[] = {
        1, 4, 7,
        2, 5, 8, 
        3, 6, 9
    };
    float testarray2[] = {
        2, 1, 4,
        0, 2, 1,
        1, 3, 2
    };

    // Helper function to compare arrays
    auto compare_arrays = [](const float* actual, const float* expected, size_t size, 
                           float rel_tol, float abs_tol) {
        for (size_t i = 0; i < size; ++i) {
            if (!check_result(actual[i], expected[i], rel_tol, abs_tol)) {
                return ::testing::AssertionFailure() 
                    << "Arrays differ at index " << i 
                    << ". Expected: " << expected[i] 
                    << ", Actual: " << actual[i];
            }
        }
        return ::testing::AssertionSuccess();
    };

    // CPU tests
    result_matrix = cpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix1, rows1, cols1);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray1, array1_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;

    result_matrix = cpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix2, rows1, cols1);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray2, array1_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;

    // GPU tests
    result_matrix = gpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix1, rows1, cols1);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray1, array1_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;

    result_matrix = gpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix2, rows1, cols1);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray2, array1_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;


    size_t array2_size = 16;

	float testarray3[] = {
		5, 12, 19, 3.1,
		9.7, 91, 74, 82, 
		1, 15, 3.2, 31,
		6.2, 4.7, 9.1, 22
	};

	float testarray4[] = {
		7.5, 11.2, 17.6, 1.5, 
		2.3, 5.9, 6.4, 9.2, 
		8.1, 14.3, 2.1, 12.7, 
		4.6, 3.8, 8.9, 4.3
	};
    result_matrix = cpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix3, rows2, cols2);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray3, array2_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;

    result_matrix = cpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix4, rows2, cols2);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray4, array2_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;

    // GPU tests
    result_matrix = gpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix3, rows2, cols2);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray3, array2_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;

    result_matrix = gpuopmanager->single_vector_op(operation_types::TRANSPOSE, matrix4, rows2, cols2);
    EXPECT_TRUE(compare_arrays(result_matrix, testarray4, array2_size, relative_tolerance, absolute_tolerance));
    delete[] result_matrix;



}

TEST_F(OperationTest, Inverse_Test)
{
}
