// CUDA Kernel to shift pointers in a GPU array
extern "C" __global__ void pointerShiftKernel(double **from, int fromInc, double** to, int toInc, int n, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shift each pointer by the given right and down offsets
    if(idx < n){
        to[idx*toInc] = from[idx*fromInc] + shift;
    }
}

