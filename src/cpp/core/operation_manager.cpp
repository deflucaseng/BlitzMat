#include "include/operation_manager.hpp"

OperationManager::OperationManager(device_types device_type){

    clGetPlatformIDs(1, &platform, &num_platforms);

	switch(device_type){
		case(device_types::CPU_DEVICE):
    		clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);

		case(device_types::GPU_DEVICE):
    		clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);

	}
    
    // Step 2: Create Context and Command Queue
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);
}


OperationManager::~OperationManager(){
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}


double* OperationManager::multi_vector_op(operation_types op_type, double* lhs, int lheight, int lwidth, double* rhs, int rheight, int rwidth){
    cl_int err;
    
    // Create program
    const char* kernel_source = *kernel_manager.getKernelSource(op_type);
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program");
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log for debugging
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        clReleaseProgram(program);
        throw std::runtime_error("Failed to build program: " + std::string(build_log.data()));
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "blitz_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Failed to create kernel");
    }



	const size_t lhs_size = lheight * lwidth * sizeof(double);
	const size_t rhs_size = rheight * rwidth * sizeof(double);
	
	double* matrix_result = nullptr;
	cl_mem lhs_buffer = nullptr;
	cl_mem rhs_buffer = nullptr;
	cl_mem result_buffer = nullptr;

	// Create buffers
	lhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								lhs_size, lhs, &err);
	rhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
							rhs_size, rhs, &err);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to create input buffer");
	}


	size_t result_size;
	try{
    	switch(op_type) {
			case operation_types::ELEM_WISE_ADD:
			case operation_types::ELEM_WISE_SUB:
			case operation_types::ELEM_WISE_MUL:
			case operation_types::ELEM_WISE_DIV:{
                // Allocate host memory
				result_size = lhs_size;
				break;
        	}
			case operation_types::MATRIX_MULTIPLICATION:{
                // Allocate host memory
				result_size = lheight * rwidth * sizeof(double);
				break;
        	}
			default:
				throw std::runtime_error("Incorrect Operation Type");

		}	
    } catch (...) {
		if (lhs_buffer) clReleaseMemObject(lhs_buffer);
		if (rhs_buffer) clReleaseMemObject(rhs_buffer);
		if (result_buffer) clReleaseMemObject(result_buffer);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		throw;
            
    }

	matrix_result = (double*)malloc(result_size);
	result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, result_size, NULL, &err);
	if (!matrix_result) {
		throw std::bad_alloc();
	}
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to create result buffer");
	}

	// Set kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &lhs_buffer);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &rhs_buffer);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &lheight);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &lwidth);
	err |= clSetKernelArg(kernel, 5, sizeof(int), &rheight);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &rwidth);

	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to set kernel arguments");
	}
	
	// Execute kernel
	size_t global_work_size[2] = {static_cast<size_t>(lheight), static_cast<size_t>(rwidth)};
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to execute kernel");
	}

	// Read results
	err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, lheight * rwidth * sizeof(double), matrix_result, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to read results");
	}

		// Cleanup
	clReleaseMemObject(lhs_buffer);
	clReleaseMemObject(rhs_buffer);
	clReleaseMemObject(result_buffer);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	return matrix_result;
}






double* OperationManager::single_vector_op(operation_types op_type, double* data, int height, int width) {
    cl_int err;
	if (op_type == operation_types::DETERMINANT || op_type == operation_types::INVERSE) {
		if (height != width) {
			throw std::invalid_argument("Operation requires square matrix");
		}
	}
    // Create program
    const char* kernel_source = *kernel_manager.getKernelSource(op_type);
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program");
    }

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log for debugging
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        clReleaseProgram(program);
        throw std::runtime_error("Failed to build program: " + std::string(build_log.data()));
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "blitz_kernel", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        throw std::runtime_error("Failed to create kernel");
    }



	const size_t input_size = height * width * sizeof(double);
	
	double* matrix_result = nullptr;
	cl_mem input_buffer = nullptr;
	cl_mem result_buffer = nullptr;

	// Create buffers
	input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
								input_size, data, &err);

	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to create input buffer");
	}
	size_t output_size;
	try{
    	switch(op_type) {
			case operation_types::DETERMINANT:
			case operation_types::FROBENIUS_NORM:
			case operation_types::TRACE:{
				output_size = sizeof(double);
                // Allocate host memory

        	}
			case operation_types::INVERSE:
			case operation_types::TRANSPOSE:{
				output_size = sizeof(double) * height * width;
                // Allocate host memory

        	}
			default:
				throw std::runtime_error("Incorrect Operation Type");

		}	
    } catch (...) {
		if (input_buffer) clReleaseMemObject(input_buffer);
		if (result_buffer) clReleaseMemObject(result_buffer);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		throw;
            
    }

	matrix_result = (double*)malloc(output_size);
	result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
								output_size, NULL, &err);

	if (!matrix_result) {
		throw std::bad_alloc();
	}
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to create result buffer");
	}

	// Set kernel arguments
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &result_buffer);
	err |= clSetKernelArg(kernel, 2, sizeof(int), &height);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &width);

	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to set kernel arguments");
	}
	
	size_t global_work_size[2];
	if (op_type == operation_types::DETERMINANT || 
		op_type == operation_types::FROBENIUS_NORM || 
		op_type == operation_types::TRACE) {
		global_work_size[0] = 1;
		global_work_size[1] = 1;
	} else {
		global_work_size[0] = static_cast<size_t>(height);
		global_work_size[1] = static_cast<size_t>(width);
	}






	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to execute kernel");
	}

	// Read results
	err = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, output_size, matrix_result, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to read results");
	}

		// Cleanup
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(result_buffer);
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	return matrix_result;
}