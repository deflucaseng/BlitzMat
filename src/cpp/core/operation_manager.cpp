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

	cl_program program = clCreateProgramWithSource(context, 1, kernel_manager.getKernelSource(op_type), NULL, NULL);
	clBuildProgram(program, 1, &device, NULL, NULL, NULL);


	cl_kernel kernel = clCreateKernel(program, "blitz_kernel", NULL);

	switch(op_type){
		case operation_types::ELEM_WISE_ADD:
		case operation_types::ELEM_WISE_SUB:
		case operation_types::ELEM_WISE_DIV:
		case operation_types::ELEM_WISE_MUL:
			assert(lheight == rheight);
			assert(lwidth == rwidth);

			int lhs_size = lheight * lwidth * sizeof(double);
			int rhs_size = rheight * rwidth * sizeof(double);

			double* matrix_result = (double*) malloc(lhs_size);

			cl_mem lhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										lhs_size, lhs, NULL);
			cl_mem rhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										rhs_size, rhs, NULL);
			cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										lhs_size, NULL, NULL);	

			clSetKernelArg(kernel, 0, sizeof(cl_mem), &lhs_buffer);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &rhs_buffer);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
			clSetKernelArg(kernel, 3, sizeof(int), &lheight);
			clSetKernelArg(kernel, 4, sizeof(int), &lwidth);

			size_t global_work_size[2] = {lheight, lwidth};
			clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

			clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, lhs_size, matrix_result, 0, NULL, NULL);


			clReleaseMemObject(lhs_buffer);
			clReleaseMemObject(rhs_buffer);
			clReleaseMemObject(result_buffer);
			
			break;


		case operation_types::MATRIX_MULTIPLICATION:
			assert(lwidth == rheight);

			int lhs_size = lheight * lwidth * sizeof(double);
			int rhs_size = rheight * rwidth * sizeof(double);

			double* matrix_result = (double*) malloc(lheight * rwidth);

			cl_mem lhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										lhs_size, lhs, NULL);
			cl_mem rhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
										rhs_size, rhs, NULL);
			cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
										lheight * rwidth, NULL, NULL);	

			clSetKernelArg(kernel, 0, sizeof(cl_mem), &lhs_buffer);
			clSetKernelArg(kernel, 1, sizeof(cl_mem), &rhs_buffer);
			clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_buffer);
			clSetKernelArg(kernel, 3, sizeof(int), &lheight);
			clSetKernelArg(kernel, 4, sizeof(int), &lwidth);
			clSetKernelArg(kernel, 5, sizeof(int), &rwidth);

			size_t global_work_size[2] = {lheight, rwidth};
			clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

			clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, lhs_size, matrix_result, 0, NULL, NULL);


			clReleaseMemObject(lhs_buffer);
			clReleaseMemObject(rhs_buffer);
			clReleaseMemObject(result_buffer);
			
			break;

		default:
			throw std::runtime_error("Incorrect Operation Type");
		
	}


}



double* OperationManager::single_vector_op(operation_types op_type, double* data, int height, int width){

}

