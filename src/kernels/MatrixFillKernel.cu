extern "C"
__global__ void fillMatrix(double* A, int height, int width, int lda, double scalar) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width)
        A[row + col * lda] = scalar;
    
}

//turn into executable with: nvcc -arch=compute_75 -code=sm_75,compute_75 MatrixFillKernel.cu -ptx -o MatrixFillKernel.ptx

