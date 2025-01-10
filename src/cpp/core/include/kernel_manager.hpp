#ifndef KERNEL_MANAGER_HPP
#define KERNEL_MANAGER_HPP

#include <unordered_map>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include "operation_types.hpp"
class KernelManager{
	public:
		class KernelError : public std::runtime_error {
			using std::runtime_error::runtime_error;
		};


		const char** getKernelSource(operation_types binding_name) const;

	private:
		std::unordered_map<operation_types, std::string> lookup_table = {

		//	Name used in Binding						File Location
			{operation_types::DETERMINANT, 				"kernels/determinant.cl"},
			{operation_types::ELEM_WISE_ADD, 			"kernels/elem_add.cl"},
			{operation_types::ELEM_WISE_DIV, 			"kernels/elem_div.cl"},
			{operation_types::ELEM_WISE_MUL, 			"kernels/elem_mul.cl"},
			{operation_types::ELEM_WISE_SUB, 			"kernels/elem_sub.cl"},
			{operation_types::INVERSE,					"kernels/inverse.cl"},
			{operation_types::TRACE,					"kernels/trace.cl"},
			{operation_types::TRANSPOSE,				"kernels/transpose.cl"},
			{operation_types::MATRIX_MULTIPLICATION, 	"kernels/mat_mul.cl"},
			{operation_types::FROBENIUS_NORM, 			"kernels/frb_nrm.cl"}
		};
		mutable std::unordered_map<operation_types, std::string> kernel_sources;
		mutable const char* current_source;
    	mutable const char* source_array[1];  // Array of size 1 for OpenCL
};



#endif