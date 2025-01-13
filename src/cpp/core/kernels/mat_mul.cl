__kernel void blitz_kernel(
    __global const float* lhs,     // First input matrix
    __global const float* rhs,     // Second input matrix
    __global float* result,        // Output matrix
    const int lheight,             // Height of first matrix
    const int lwidth,              // Width of first matrix
    const int rheight,             // Height of second matrix
    const int rwidth               // Width of second matrix
) {
    // Get global position in the result matrix
    int row = get_global_id(0);    // Row index
    int col = get_global_id(1);    // Column index
    
    // Check if we're within bounds
    if (row < lheight && col < rwidth) {
        float sum = 0.0f;
        
        // Perform dot product of row from lhs and column from rhs
        for (int k = 0; k < lwidth; k++) {
            float lhs_element = lhs[row * lwidth + k];
            float rhs_element = rhs[k * rwidth + col];
            sum += lhs_element * rhs_element;
        }
        
        // Store the result
        result[row * rwidth + col] = sum;
    }
}