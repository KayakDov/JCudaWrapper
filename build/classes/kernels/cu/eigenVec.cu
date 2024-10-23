// LU: The matrix containing LU decomposition of (A - lambda*I)
// pivot: Pivot array from LU decomposition
// eVec: Output eigenvector, size n
// n: Size of the matrix (either 2 or 3)

//Doesn't work yet!
extern "C" __global__ void eigenVecKernel(double *LU, int ldLU, double *eVec, intldTo, int *pivot, int n) {
    

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;  

    // Forward substitution to solve L*y = 0
    __shared__ double y[3];  // Shared memory to store intermediate result y
    if (idx == 0) {
        y[0] = 1.0;  // First element of y is always 1 due to unit diagonal in L
    }
    else {
        // Perform forward substitution to compute y[idx]
        double sum = 0.0;
        for (int j = 0; j < idx; ++j) {
            sum += LU[idx * n + j] * y[j];
        }
        y[idx] = -sum;  // Since we're solving (L*y = 0), negate the sum
    }

    // Synchronize to ensure all threads have computed y
    __syncthreads();

    // Backward substitution to solve U*x = y
    double x = y[idx];  // Start with the current value of y[idx]

    for (int j = n - 1; j > idx; --j) {
        x -= LU[idx * n + j] * eVec[j];  // Subtract the known terms
    }

    // Divide by the diagonal element of U
    eVec[idx] = x / LU[idx * n + idx];  // Store the result as the eigenvector
}

