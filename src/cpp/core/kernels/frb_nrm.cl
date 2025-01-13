__kernel void blitz_kernel(__global const float* input, __global float* result,
                            const int height, const int width) {
    float sum = 0.0;
    
    // First work item calculates the sum of squares
    if (get_global_id(0) == 0) {
        for (int i = 0; i < height * width; i++) {
            sum += input[i] * input[i];
        }
        *result = sqrt(sum);
    }
}