// CUDA Kernel to shift pointers in a GPU array
extern "C" __global__ void pointerShiftKernel(double **from, int inc, double** notUsed, int shift, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shift each pointer by the given right and down offsets
    if(idx < n){
        from[idx*inc] += shift;
    }
}

