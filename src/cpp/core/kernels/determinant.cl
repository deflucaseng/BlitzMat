__kernel void blitz_kernel(__global const float* input, __global float* result,
                         const int height, const int width) {
    if (get_global_id(0) == 0) {
        if (height != width) {
            *result = 0.0f;
            return;
        }

        int n = height;
        float L[16][16];
        float U[16][16];
        
        // Initialize with explicit float literals
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                if(i == j)
                    L[i][i] = 1.0f;
                else
                    L[i][j] = 0.0f;
                U[i][j] = 0.0f;
            }
        }
        
        const float EPSILON = 1.0e-6f;  // Threshold for zero
        
        for(int i = 0; i < n; i++) {
            // Upper triangular matrix U
            for(int k = i; k < n; k++) {
                float sum = 0.0f;
                // Use Kahan summation for better precision
                float c = 0.0f;  // Running compensation
                for(int j = 0; j < i; j++) {
                    float y = L[i][j] * U[j][k] - c;
                    float t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                U[i][k] = input[i * n + k] - sum;
            }
            
            // Check for numerical stability
            if(fabs(U[i][i]) < EPSILON) {
                *result = 0.0f;
                return;
            }
            
            // Lower triangular matrix L
            for(int k = i + 1; k < n; k++) {
                float sum = 0.0f;
                float c = 0.0f;  // Kahan summation compensation
                for(int j = 0; j < i; j++) {
                    float y = L[k][j] * U[j][i] - c;
                    float t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                L[k][i] = (input[k * n + i] - sum) / U[i][i];
            }
        }
        
        // Calculate determinant with log-sum-exp trick for better numerical stability
        float logdet = 0.0f;
        for(int i = 0; i < n; i++) {
            logdet += log(fabs(U[i][i]));
        }
        *result = exp(logdet);
        
        // Adjust sign
        int sign = 1;
        for(int i = 0; i < n; i++) {
            if(U[i][i] < 0.0f) sign = -sign;
        }
        *result *= sign;
    }
}