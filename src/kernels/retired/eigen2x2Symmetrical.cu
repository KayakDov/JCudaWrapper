__global__ void eigen2x2Kernel(double *matrices, int num_matrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_matrices) {
        // Each matrix has 3 elements per column with lda = 3 (column-major order)
        // Matrix layout: [a, b, lambda1, b, d, lambda2]
        double a = matrices[idx * 6];      // matrix[0][0]
        double b = matrices[idx * 6 + 1];  // matrix[1][0]
        double d = matrices[idx * 6 + 4];  // matrix[1][1]

        // Compute the eigenvalues using the standard formula
        double halfTrace = (a + d)/2;
        double det = a * d - b * b;
        double sqrt_term = sqrt(halfTrace*halfTrace - det);

        // Eigenvalues
        matrices[idx * 6 + 2] = halfTrace + sqrt_term;  // First eigenvalue
        matrices[idx * 6 + 5] = halfTrace - sqrt_term;  // Second eigenvalue

	// Compute the eigenvectors
        // Eigenvector for lambda1: (v1x, v1y)
        double v1x, v1y;

        if (b != 0) {
            v1x = matrices[idx * 6 + 2] - d;
            v1y = b;
        } else {
            v1x = 1;
            v1y = 0;  // When b = 0, this eigenvector is aligned with the x-axis
        }

        // Store the first eigenvector back in the matrix (column-major order)
        matrices[idx * 6] = v1x;      // Update matrix[0][0] with v1x
        matrices[idx * 6 + 1] = v1y;  // Update matrix[1][0] with v1y

        // The second eigenvector (v2x, v2y) is orthogonal to the first
        double v2x = -v1y;
        double v2y = v1x;

        // Store the second eigenvector in the matrix (column-major order)
        matrices[idx * 6 + 3] = v2x;  // Update matrix[0][1] with v2x (effectively matrix[0][1] = -v1y)
        matrices[idx * 6 + 4] = v2y;  // Update matrix[1][1] with v2y (effectively matrix[1][1] = v1x)
    }
}
//to compile nvcc -ptx eigen2x2Symmetrical.cu 

