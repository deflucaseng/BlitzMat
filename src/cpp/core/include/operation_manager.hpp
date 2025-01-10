#ifndef OPERATION_MANAGER_HPP
#define OPERATION_MANAGER_HPP

#include <string>
#include <CL/cl.h>

#include "kernel_manager.hpp"
#include "operation_types.hpp"
#include <cassert>

class OperationManager{
	public:
		enum class device_types{
			CPU_DEVICE,
			GPU_DEVICE
		};
		OperationManager(device_types device_type); // Sets context/queue
		~OperationManager(); //Releases Queue/Context

		double* multi_vector_op(operation_types op_type, double* lhs, int lheight, int lwidth, double* rhs, int rheight, int rwidth);
		double* single_vector_op(operation_types op_type, double* data, int height, int width);





	private:
		KernelManager kernel_manager;

		cl_platform_id platform;
		cl_device_id device;
		cl_uint num_platforms, num_devices;
		cl_context context;
		cl_command_queue queue;
};




#endif