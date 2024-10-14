//The first value of from us used to fill this matrix.
//n is the size of the matrix, width * height.

extern "C" __global__ void fillMatrixKernel(double *from, int lda, double *to, int height, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the index is within bounds
    if (idx < n) {
        to[(idx/height)*lda + idx % height] = from[0]; 
    }
}
//nvcc -ptx fillMatrix.cu 