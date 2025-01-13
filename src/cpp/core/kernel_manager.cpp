#include "include/kernel_manager.hpp"

const char** KernelManager::getKernelSource(operation_types binding_name) const {
    auto location = lookup_table.find(binding_name);
    if (location == lookup_table.end()) {
        throw std::runtime_error("Kernel binding name not found");
    }

    // Check if we already have the kernel source loaded
    auto existing_source = kernel_sources.find(binding_name);
    if (existing_source == kernel_sources.end()) {
        // Only load the file if we haven't already
        std::ifstream file(location->second);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open kernel file: " + location->second);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();

        // Store the source
        kernel_sources.insert({binding_name, buffer.str()});
    }

    // Update the pointer to the persistent string data
    current_source = kernel_sources.at(binding_name).c_str();
    source_array[0] = current_source;

    return source_array;
}