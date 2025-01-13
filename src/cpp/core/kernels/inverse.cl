__kernel void blitz_kernel(
    __global const float* input,   // Input matrix
    __global float* result,        // Output matrix
    const int height,              // Height of input matrix
    const int width               // Width of input matrix
) {
    int row = get_global_id(0);
    
    // Each work item processes one row
    if (row < height) {
        // First, copy input to result and create augmented matrix
        for (int col = 0; col < width; col++) {
            // Copy original matrix
            result[row * 2*width + col] = input[row * width + col];
            // Create identity matrix in augmented portion
            result[row * 2*width + width + col] = (row == col) ? 1.0f : 0.0f;
        }
        
        barrier(CLK_GLOBAL_MEM_FENCE);
        
        // For each pivot element (diagonal)
        for (int pivot = 0; pivot < height; pivot++) {
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            // If this is the pivot row
            if (row == pivot) {
                // Normalize the pivot row
                float pivot_val = result[row * 2*width + pivot];
                if (fabs(pivot_val) > 1e-10) {  // Check if not too close to zero
                    for (int col = 0; col < 2*width; col++) {
                        result[row * 2*width + col] /= pivot_val;
                    }
                }
            }
            
            barrier(CLK_GLOBAL_MEM_FENCE);
            
            // Eliminate pivot element from other rows
            if (row != pivot) {
                float factor = result[row * 2*width + pivot];
                for (int col = 0; col < 2*width; col++) {
                    result[row * 2*width + col] -= factor * result[pivot * 2*width + col];
                }
            }
        }
    }
}