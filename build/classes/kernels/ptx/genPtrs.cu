// CUDA Kernel to generate pointers in a GPU array
extern "C" __global__ void genPtrsKernel(double **array, int inc, double** firstPointer, int shift, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shift each pointer by the given right and down offsets
    if(idx < n){
        if(idx == 0) array[0] = firstPointer[0];
        else array[idx*inc] = array[idx*(inc - 1)] + shift;
    }
}

