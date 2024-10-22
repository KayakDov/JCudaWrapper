package main;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;

/**
 *
 * @author edov
 */
public class testLUSolver {
    
    public static void main(String[] args) {
        

        // Allocate and initialize host memory for the 2x2 matrix (row-major format)
        double[] h_A = {1.0, 2.0, 3.0, 4.0};  // A = [1 2; 3 4]
        double[] h_B = {1.0, 2.0};            // B = [1; 2], right-hand side
        int[] h_PivotArray = new int[2];      // Pivot array for LU factorization
        int[] h_InfoArray = new int[1];       // Info array for success check

        // Allocate device memory
        Pointer d_A = new Pointer();
        Pointer d_B = new Pointer();
        Pointer d_PivotArray = new Pointer();
        Pointer d_InfoArray = new Pointer();
        
        JCuda.cudaMalloc(d_A, 4 * Sizeof.DOUBLE);        // 2x2 matrix A
        JCuda.cudaMalloc(d_B, 2 * Sizeof.DOUBLE);        // B vector
        JCuda.cudaMalloc(d_PivotArray, 2 * Sizeof.INT);  // Pivot array
        JCuda.cudaMalloc(d_InfoArray, 1 * Sizeof.INT);   // Info array

        // Copy host memory to device
        JCublas2.cublasSetVector(4, Sizeof.DOUBLE, Pointer.to(h_A), 1, d_A, 1);
        JCublas2.cublasSetVector(2, Sizeof.DOUBLE, Pointer.to(h_B), 1, d_B, 1);
        JCublas2.cublasSetVector(2, Sizeof.INT, Pointer.to(h_PivotArray), 1, d_PivotArray, 1);
        
        // Create handle
        cublasHandle handle = new cublasHandle();
        JCublas2.cublasCreate(handle);

        // LU factorization (in-place)
        JCublas2.cublasDgetrfBatched(handle, 2, Pointer.to(d_A), 2, d_PivotArray, d_InfoArray, 1);

        // Solve system Ax = B using the LU factorization
        JCublas2.cublasDgetrsBatched(handle, cublasOperation.CUBLAS_OP_N, 2, 1, Pointer.to(d_A), 2, d_PivotArray, Pointer.to(d_B), 2, d_InfoArray, 1);

        // Copy result back to host
        JCublas2.cublasGetVector(2, Sizeof.DOUBLE, d_B, 1, Pointer.to(h_B), 1);

        // Print solution
        System.out.println("Solution x:");
        for (int i = 0; i < 2; i++) {
            System.out.println(h_B[i]);
        }

        // Clean up
        JCuda.cudaFree(d_A);
        JCuda.cudaFree(d_B);
        JCuda.cudaFree(d_PivotArray);
        JCuda.cudaFree(d_InfoArray);
        JCublas2.cublasDestroy(handle);
    }
}
