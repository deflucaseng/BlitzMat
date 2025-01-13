__kernel void blitz_kernel(
    __global const float* input,   // Input matrix
    __global float* result,        // Output matrix
    const int height,              // Height of input matrix
    const int width               // Width of input matrix
) {
    int row = get_global_id(0);    // Row index
    int col = get_global_id(1);    // Column index
    
    // Check bounds
    if (row < width && col < height) {  // Note: swapped height/width in check
        // Transpose by reading from input[row + col * width] 
        // and writing to result[col + row * height]
        result[col + row * height] = input[row + col * width];
    }
}