#include "include/kernel_manager.hpp"

const char** KernelManager::getKernelSource(operation_types binding_name) const {

	
	auto location = lookup_table.find(binding_name);
	if (location == lookup_table.end()) {
		throw std::runtime_error("Kernel binding name not found");
	}

	std::ifstream file(location->second);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open kernel file: " + location->second);
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	
	kernel_sources[binding_name] = buffer.str();
	
	// Store the pointer in the array
	current_source = kernel_sources[binding_name].c_str();
	source_array[0] = current_source;
	
	return source_array;
}