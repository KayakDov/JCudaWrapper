
extern "C" __global__ void sqrtKernel(double *from, int incFrom, double *to, int incTo, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < n) {
        to[idx * incTo] = sqrt(from[idx * incFrom]); 
    }
}
//nvcc -ptx sqrt.cu 