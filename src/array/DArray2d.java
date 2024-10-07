package array;

import java.util.Arrays;
import java.util.function.IntUnaryOperator;
import jcuda.Pointer;
import jcuda.cudaDataType;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasGemmAlgo;
import jcuda.jcublas.cublasOperation;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverDnHandle;
import jcuda.jcusolver.cusolverEigMode;
import jcuda.jcusolver.syevjInfo;
import org.apache.commons.math3.exception.DimensionMismatchException;
import resourceManagement.Handle;
import static array.Array.checkNull;
import static array.Array.checkPos;
import static array.DArray.cpuPointer;
import jcuda.jcublas.cublasFillMode;

/**
 * Class for managing a batched 2D array of arrays (DArrays) on the GPU and
 * supporting various operations including batched matrix-matrix multiplication.
 *
 * Provides support for batched matrix-matrix multiplication using the cuBLAS
 * library.
 *
 * @author E. Dov Neimand
 */
public class DArray2d extends Array {

    private final int lengthOfArrays;
    
    
    
    /**
     * An array of Arrays.
     *
     * @param p The pointer to this array.
     * @param lengthOfArrays The length of the arrays.
     * @param numberOfArrays The number of arrays stored in this array. The
     * length of this array of arrays.
     */
    private DArray2d(CUdeviceptr p, int lengthOfArrays, int numberOfArrays) {
        super(p, numberOfArrays, PrimitiveType.POINTER);
        this.lengthOfArrays = lengthOfArrays;
    }

    /**
     * Stores the list of pointers in the gpu.
     *
     * @param handle The handle
     * @param arrays The arrays to be stored in this array. This array must be
     * nonempty.
     */
    public DArray2d(Handle handle, DArray[] arrays) {
        super(empty(arrays.length, PrimitiveType.POINTER), arrays.length, PrimitiveType.POINTER);
        Pointer[] pointers = Arrays.stream(arrays).map(a -> a.pointer).toArray(Pointer[]::new);
        set(handle, Pointer.to(pointers), length);
        lengthOfArrays = arrays.length;
    }

    /**
     * Creates an empty DArray with the specified size.
     *
     * @param length The number of elements in the array.
     * @param lengthOfArrays The length of the DArrays stored in the new array.
     * @return A new DArray with the specified size.
     * @throws ArrayIndexOutOfBoundsException if size is negative.
     */
    public static DArray2d empty(int length, int lengthOfArrays) {
        checkPos(length);
        return new DArray2d(Array.empty(length, PrimitiveType.POINTER), lengthOfArrays, length);
    }

    /**
     * Sets an index of this array.
     *
     * @param handle The handle.
     * @param array The array to be placed at the given index.
     * @param index The index to place the array at.
     */
    public void set(Handle handle, DArray array, int index) {
        if (array.length != lengthOfArrays)
            throw new DimensionMismatchException(array.length, lengthOfArrays);
        super.set(handle, array.pointer, index);
    }

    /**
     * Gets the array at the given index from GPU and transfers it to CPU
     * memory.
     *
     * @param handle The handle.
     * @param index The index of the desired array.
     * @return The array at the given index.
     */
    public DArray get(Handle handle, int index) {
        checkPos(index);
        checkAgainstLength(index);

        CUdeviceptr[] hostPointer = new CUdeviceptr[1];

        get(handle, Pointer.to(hostPointer), 0, index, 1);

        return new DArray(hostPointer[0], lengthOfArrays);
    }

    /**
     * Sets the elements of this array to be pointers to sub sections of the
     * proffered array.
     *
     * @param handle The Handle.
     * @param source An array with sub arrays that are held in this array.
     * @param generator The index of the beginning of the sub array to be held
     * at the argument's index.
     * @return This.
     */
    public DArray2d set(Handle handle, DArray source, IntUnaryOperator generator) {
        CUdeviceptr[] pointers = new CUdeviceptr[length];
        Arrays.setAll(pointers, i -> pointer(generator.applyAsInt(i)));
        set(handle, Pointer.to(pointers), 0, length);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray2d copy(Handle handle) {
        DArray2d copy = empty(length, lengthOfArrays);
        get(handle, copy, 0, 0, length);
        return copy;
    }

    /**
     * Performs batched matrix-matrix multiplication:
     *
     * <pre>
     * Result[i] = timesAB * op(A[i]) * op(B[i]) + timesResult * this[i]
     * </pre>
     *
     * Where op(A) and op(B) can be A and B or their transposes.
     *
     * This method computes multiple matrix-matrix multiplications at once
     * without using strided data access, i.e., it processes independent
     * batches.
     *
     * @param handle Handle to the cuBLAS library context.
     * @param transA True if matrix A should be transposed, false otherwise.
     * @param transB True if matrix B should be transposed, false otherwise.
     * @param aRows The number of rows in matrix A.
     * @param aColsBRows The number of columns in matrix A and the number of
     * rows in matrix B.
     * @param bCols The number of columns in matrix B.
     * @param timesAB Scalar multiplier applied to the matrix-matrix product.
     * @param A Array of pointers to matrices A (in GPU memory).
     * @param lda Leading dimension of each matrix A (number of elements between
     * consecutive columns in memory).
     * @param B Array of pointers to matrices B (in GPU memory).
     * @param ldb Leading dimension of each matrix B (number of elements between
     * consecutive columns in memory).
     * @param timesResult Scalar multiplier applied to each result matrix before
     * adding the matrix-matrix product.
     * @param ldResult Leading dimension of each result matrix (number of
     * elements between consecutive columns in memory).
     * @param batchCount The number of matrix-matrix multiplications to compute.
     */
    public void multMatMatBatched(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DArray2d A,
            int lda, DArray2d B, int ldb, double timesResult, int ldResult, int batchCount) {

        checkNull(handle, A, B);
        checkPos(aRows, aColsBRows, bCols, batchCount);
        checkLowerBound(aRows, lda, ldResult);
        checkLowerBound(aColsBRows, ldb);

        // Perform the batched matrix-matrix multiplication using cuBLAS
        JCublas2.cublasDgemmBatched(
                handle.get(), // cuBLAS handle
                DArray.transpose(transA),
                DArray.transpose(transB),
                aRows, bCols, aColsBRows, // Number of columns of A / rows of B
                DArray.cpuPointer(timesAB),
                A.pointer, lda, // Leading dimension of A
                B.pointer, ldb, // Leading dimension of B
                DArray.cpuPointer(timesResult), pointer, ldResult, // Leading dimension of result matrices
                batchCount // Number of matrices to multiply
        );
    }

    /**
     * Fill modes.  Use lower to indicate a lower triangle, upper to indicate an upper triangle, and full for full triangles.
     */
    public static enum Fill {
        LOWER(cublasFillMode.CUBLAS_FILL_MODE_LOWER), UPPER(cublasFillMode.CUBLAS_FILL_MODE_UPPER), FULL(cublasFillMode.CUBLAS_FILL_MODE_FULL);

        private int fillMode;

        private Fill(int fillMode) {
            this.fillMode = fillMode;
        }

        public int getFillMode() {
            return fillMode;
        }

    }

    /**
     * https://docs.nvidia.com/cuda/cusolver/index.html?highlight=cusolverDnCheevjBatched#cuSolverDN-lt-t-gt-syevjbatch
     *
     * Computes the eigenvalues and eigenvectors of a batch of symmetric
     * matrices using the cuSolver library.
     *
     * This method leverages the cusolverDnDsyevjBatched function, which
     * computes the eigenvalues and eigenvectors of symmetric matrices using the
     * Jacobi method.
     *
     * This method creates and destroys it's own handle since it uses a
     * different sort of handle then the handle class.
     *
     * @param height The height and width of the matrices.
     * @param inputMatrices The input matrices, which must be stored in GPU
     * memory consecutively so that each matrix is column-major with leading
     * dimension lda, so the formula for random access is a_k[i, j] = A[i +
     * lda*j + lda*n*k]
     * @param ldInput The leading dimension of the input matrices.
     * @param resultValues Array to store the eigenvalues of the matrices.
     * @param resultVectors Array to store the eigenvectors of the matrices.
     * @param ldResultVectors The leading dimension of the result vectors. This
     * is a matrix, each column is an eigan vector.
     * @param batchCount The number of matrices in the batch.
     * @param cublasFillMode Fill mode for the symmetric matrix (upper or
     * lower).
     */
    public static void computeEigen(int height, DArray inputMatrices, int ldInput,
            DArray resultValues, DArray resultVectors, int ldResultVectors,
            int batchCount, Fill cublasFillMode) {

        cusolverDnHandle solverHandle = new cusolverDnHandle();
        JCusolverDn.cusolverDnCreate(solverHandle); // Create handle

        syevjInfo params = new syevjInfo(); // Correct type for Jacobi parameters
        JCusolverDn.cusolverDnCreateSyevjInfo(params); // Create parameter structure
        try (IArray info = IArray.empty(batchCount)) {

            JCusolverDn.cusolverDnDsyevjBatched(
                    solverHandle, // Handle to the cuSolver context
                    cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR, // Compute both eigenvalues and eigenvectors
                    cublasFillMode.getFillMode(), // Indicates matrix is symmetric
                    height, // Size of each matrix
                    inputMatrices.pointer, // Pointer to input matrices in GPU memory
                    ldInput, // Leading dimension of input matrices
                    resultValues.pointer, // Pointer to output array for eigenvalues
                    resultVectors.pointer, // Pointer to output array for eigenvectors
                    ldResultVectors, // Leading dimension of result vectors
                    info.pointer, // Array to store status info for each matrix
                    params, // Jacobi algorithm parameters
                    batchCount // Number of matrices in the batch
            );

            // Step 5: Check for convergence status in info array
            int[] infoHost = new int[batchCount]; // Host array to fetch status

            Handle hand = new Handle();
            info.get(hand, infoHost, 0, 0, batchCount);
            hand.close();

            for (int i = 0; i < batchCount; i++) {
                if (infoHost[i] != 0) {
                    System.err.println("Matrix " + i + " failed to converge: info = " + infoHost[i]);
                }
            }

            // Step 6: Clean up resources
            JCusolverDn.cusolverDnDestroySyevjInfo(params);
            JCusolverDn.cusolverDnDestroy(solverHandle);     // Destroy solver handle
        }
    }

    /**
     * Performs batched matrix-matrix multiplication:
     *
     * <pre>
     * Result[i] = alpha * op(A[i]) * op(B[i]) + timesResult * Result[i]
     * </pre>
     *
     * Where op(A) and op(B) can be A and B or their transposes.
     *
     * This method computes multiple matrix-matrix multiplications at once,
     * using strided data access, allowing for efficient batch processing.
     *
     * @param handle Handle to the cuBLAS library context.
     * @param transA True if matrix A should be transposed, false otherwise.
     * @param transB True if matrix B should be transposed, false otherwise.
     * @param aRows The number of rows in matrix A.
     * @param aColsBRows The number of columns in matrix A and the number of
     * rows in matrix B.
     * @param bCols The number of columns in matrix B.
     * @param timesAB Scalar multiplier applied to the matrix-matrix product.
     * @param matA Pointer to the batched matrix A in GPU memory.
     * @param lda Leading dimension of matrix A (the number of elements between
     * consecutive columns in memory).
     * @param strideA Stride between consecutive matrices A in memory (number of
     * elements).
     * @param matB Pointer to the batched matrix B in GPU memory.
     * @param ldb Leading dimension of matrix B (the number of elements between
     * consecutive columns in memory).
     * @param strideB Stride between consecutive matrices B in memory (number of
     * elements).
     * @param timesResult Scalar multiplier applied to each result matrix before
     * adding the matrix-matrix product.
     * @param result Pointer to the batched output matrix (result) in GPU
     * memory.
     * @param ldResult Leading dimension of the result matrix (the number of
     * elements between consecutive columns in memory).
     * @param strideResult Stride between consecutive result matrices in memory
     * (number of elements).
     * @param batchCount The number of matrix-matrix multiplications to compute.
     *
     */
    public static void multMatMatBatched(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DArray matA,
            int lda, long strideA, DArray matB, int ldb, long strideB, double timesResult,
            DArray result, int ldResult, long strideResult, int batchCount) {

        checkNull(handle, matA, matB, result);
        checkPos(aRows, bCols, ldb, ldResult);        
        matA.checkAgainstLength(aRows * aColsBRows * batchCount - 1);
        matB.checkAgainstLength(aColsBRows * bCols * batchCount - 1);
        result.checkAgainstLength(aRows * bCols * batchCount - 1);

        
        JCublas2.cublasDgemmStridedBatched(
                handle.get(),
                DArray.transpose(transA), DArray.transpose(transB),
                aRows, bCols, aColsBRows,                
                cpuPointer(timesAB), 
                matA.pointer, lda, strideA,                
                matB.pointer, ldb, strideB,                
                cpuPointer(timesResult), result.pointer, ldResult, strideResult,                
                batchCount
        );
    }

}
