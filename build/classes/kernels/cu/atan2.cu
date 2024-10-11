extern "C" __global__
void atan2xy(const double* vectors, int ldFrom, double* angles, int incTo, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // x is at 2 * i * ldFrom, and y is at 2 * i *ldFrom + 1
        double x = vectors[ldFrom * 2 * i];
        double y = vectors[ldFrom * 2 * i + 1];
        
        // Compute the angle using atan2
        angles[incTo*i] = atan2(y, x);
    }
}
