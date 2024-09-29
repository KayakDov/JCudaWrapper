extern "C" __global__
void atan2xy(int anglesLength, const double* vectors, double* angles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < anglesLength) {
        // x is at 2 * i, and y is at 2 * i + 1
        double x = vectors[2 * i];
        double y = vectors[2 * i + 1];
        
        // Compute the angle using atan2
        angles[i] = atan2(y, x);
    }
}
