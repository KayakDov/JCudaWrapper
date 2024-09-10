#include <cuda_runtime.h>

extern "C" __global__ void fillArray(double* a, int n, double scalar, int inc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i*inc < n) a[i*inc] = scalar;
    
}


//turn into executable with: nvcc -arch=compute_75 -code=sm_75,compute_75 ArrayFillKernel.cu -ptx -o ArrayFillKernel.ptx



