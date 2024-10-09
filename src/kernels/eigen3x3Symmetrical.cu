__device__ void computeEigenvector(double a, double b, double c, double d, double e, double f, double lambda, double *matrix, int column_idx) {
    // Solves (A - lambda * I) * v = 0
    // For column-major layout, store the result in matrix starting at column_idx

    double v1, v2, v3;

    // Construct the matrix A - lambda * I
    // A - lambda * I:
    // [a - lambda, b,          c         ]
    // [b,          d - lambda, e         ]
    // [c,          e,          f - lambda]
    double a_lambda = a - lambda;
    double d_lambda = d - lambda;

    // We choose one of the equations to solve for the eigenvector
    // We can choose any two rows, but to avoid numerical instability, we pick the most non-zero coefficients

    // Use the first two rows to eliminate one variable (like Gaussian elimination)
    if (fabs(b) > 1e-6 || fabs(c) > 1e-6) {
        // Equation 1: (a - lambda) * v1 + b * v2 + c * v3 = 0
        // Equation 2: b * v1 + (d - lambda) * v2 + e * v3 = 0

        // We use these two equations to solve for v1 and v2 in terms of v3
        if (fabs(c) > 1e-6) {
            v3 = 1.0;  // Normalize v3 to 1
            v1 = -(b * e - c * d_lambda) / (c * c - a_lambda * d_lambda + b * b);
            v2 = -(a_lambda * v1 + c) / b;
        } else {
            v1 = -(d_lambda * e) / (a_lambda * b);
            v2 = 1.0;  // Normalize v2 to 1
            v3 = 0;
        }
    } else {
        // Use the third row if b and c are close to zero
        v1 = 1.0;  // Normalize v1 to 1
        v2 = 0;
        v3 = 0;
    }

    // Store the eigenvector back into the matrix in the specified column (column-major order)
    matrix[column_idx] = v1;         // matrix[0][column_idx]
    matrix[column_idx + 1] = v2;     // matrix[1][column_idx]
    matrix[column_idx + 2] = v3;     // matrix[2][column_idx]
}

__device__ void solveCubic(double p, double q, double r, double *lambda1, double *lambda2, double *lambda3) {
    // Convert the cubic equation to a depressed form: t^3 + At + B = 0
    double A = (3*q - p*p) / 3;
    double B = (2*p*p*p - 9*p*q + 27*r) / 27;
    double delta = B*B/4 + A*A*A/27;

    if (delta > 0) {
        // One real root and two complex roots (return only the real root)
        double sqrt_delta = sqrt(delta);
        double C = cbrt(-B/2 + sqrt_delta);
        double D = cbrt(-B/2 - sqrt_delta);
        *lambda1 = C + D - p / 3;
        *lambda2 = NAN;  // Complex root
        *lambda3 = NAN;  // Complex root
    } else if (delta == 0) {
        // All real roots, at least two are equal
        double C = cbrt(-B / 2);
        *lambda1 = 2 * C - p / 3;
        *lambda2 = -C - p / 3;
        *lambda3 = *lambda2;  // Double root
    } else {
        // Three real roots
        double theta = acos(-B / (2 * sqrt(-A*A*A / 27)));
        double sqrt_A = 2 * sqrt(-A / 3);
        *lambda1 = sqrt_A * cos(theta / 3) - p / 3;
        *lambda2 = sqrt_A * cos((theta + 2 * M_PI) / 3) - p / 3;
        *lambda3 = sqrt_A * cos((theta + 4 * M_PI) / 3) - p / 3;
    }
}


__global__ void eigen3x3Kernel(double *matrices, int num_matrices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_matrices) {
        // Each matrix has 4 elements per column with lda = 4 (column-major order)
        // Matrix layout:
        // Column-major storage: [a, b, c, lambda1, b, d, e, lambda2, c, e, f, lambda3]
        double a = matrices[idx * 12];         // matrix[0][0]
        double b = matrices[idx * 12 + 1];     // matrix[1][0]
        double c = matrices[idx * 12 + 2];     // matrix[2][0]
        double d = matrices[idx * 12 + 5];     // matrix[1][1]
        double e = matrices[idx * 12 + 6];     // matrix[2][1]
        double f = matrices[idx * 12 + 10];    // matrix[2][2]

        // Step 1: Compute the characteristic equation for eigenvalues.
        // For a 3x3 symmetric matrix, the cubic equation is: 
        // lambda^3 - p * lambda^2 + q * lambda - r = 0

        double trace = a + d + f;  // p = trace(A)
        double q = (a * d + a * f + d * f) - (b * b + c * c + e * e);  // sum of minors
        double r = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);  // determinant

        // Solve for the eigenvalues (lambda1, lambda2, lambda3)
        double lambda1, lambda2, lambda3;
        solveCubic(trace, q, r, &lambda1, &lambda2, &lambda3);

        // Store the eigenvalues in the 4th row (index 3, 7, 11) in column-major order
        matrices[idx * 12 + 3] = lambda1;
        matrices[idx * 12 + 7] = lambda2;
        matrices[idx * 12 + 11] = lambda3;

        // Compute and store the eigenvectors for each eigenvalue
        computeEigenvector(a, b, c, d, e, f, lambda1, matrices + idx * 12, 0);  // First eigenvector
        computeEigenvector(a, b, c, d, e, f, lambda2, matrices + idx * 12, 4);  // Second eigenvector
        computeEigenvector(a, b, c, d, e, f, lambda3, matrices + idx * 12, 8);  // Third eigenvector
    }
}

