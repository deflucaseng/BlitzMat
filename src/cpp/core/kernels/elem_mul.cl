__kernel void blitz_kernel(
    __global const float* lhs,
    __global const float* rhs,
    __global float* result,
    const int lheight,
    const int lwidth,
    const int rheight,
    const int rwidth
) {
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    if (row >= lheight || col >= lwidth) return;
    
    const int idx = row * lwidth + col;
    int rhs_row = row % rheight;
    int rhs_col = col % rwidth;
    int rhs_idx = rhs_row * rwidth + rhs_col;
    
    result[idx] = lhs[idx] * rhs[rhs_idx];
}

// Similar kernels for subtract, multiply, and divide...