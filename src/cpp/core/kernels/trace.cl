__kernel void blitz_kernel(__kernel const float* input, __global float* result,
                    const int height, const int width) {
    float sum = 0.0;
    
    // First work item calculates the trace
    if (get_global_id(0) == 0) {
        int min_dim = (height < width) ? height : width;
        for (int i = 0; i < min_dim; i++) {
            sum += input[i * width + i];  // Access diagonal elements
        }
        *result = sum;
    }
}