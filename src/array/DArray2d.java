package array;

import java.util.Arrays;
import java.util.function.IntUnaryOperator;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import org.apache.commons.math3.exception.DimensionMismatchException;
import resourceManagement.Handle;
import static array.Array.checkNull;
import static array.Array.checkPos;
import static array.DArray.cpuPointer;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasStatus;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverEigMode;
import jcuda.jcusolver.cusolverStatus;
import resourceManagement.JacobiParams;

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

        CUdeviceptr[] pointers = new CUdeviceptr[length];
        Arrays.setAll(pointers, i -> arrays[i].pointer);

        set(handle, Pointer.to(pointers), length);
        lengthOfArrays = arrays[0].length;
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
     * Creates a host array of pointer objects that have not had memory
     * allocated. These objects are ready to have actual memory addressess
     * written to them from the device.
     *
     * @param length The length of the array.
     * @return An array of new pointer objects.
     */
    private CUdeviceptr[] emptyHostArray(int length) {
        CUdeviceptr[] array = new CUdeviceptr[length];
        Arrays.setAll(array, i -> new CUdeviceptr());
        return array;
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

        CUdeviceptr[] hostPointer = emptyHostArray(1);

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
     * Fill modes. Use lower to indicate a lower triangle, upper to indicate an
     * upper triangle, and full for full triangles.
     */
    public static enum Fill {
        LOWER(cublasFillMode.CUBLAS_FILL_MODE_LOWER), UPPER(cublasFillMode.CUBLAS_FILL_MODE_UPPER), FULL(cublasFillMode.CUBLAS_FILL_MODE_FULL);

        private int fillMode;

        private Fill(int fillMode) {
            this.fillMode = fillMode;
        }

        /**
         * The fill mode integer for jcusolver methods.
         *
         * @return The fill mode integer for jcusolver methods.
         */
        public int getFillMode() {
            return fillMode;
        }

    }

//    /**
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
    public static void multMatMatStridedBatched(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DArray matA,
            int lda, int strideA, DArray matB, int ldb, int strideB, double timesResult,
            DArray result, int ldResult, long strideResult, int batchCount) {

        checkNull(handle, matA, matB, result);
        checkPos(aRows, bCols, ldb, ldResult);
        matA.checkAgainstLength(strideA * (batchCount - 1));
        matB.checkAgainstLength(strideB * (batchCount - 1));
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

    /**
     * Performs batched eigenvector computation for symmetric matrices.
     *
     * This function computes the Cholesky factorization of a sequence of
     * Hermitian positive-definite matrices.
     *
     *
     * If input parameter fill is LOWER, only lower triangular part of A is
     * processed, and replaced by lower triangular Cholesky factor L.
     *
     *
     * If input parameter uplo is UPPER, only upper triangular part of A is
     * processed, and replaced by upper triangular Cholesky factor U. * Remark:
     * the other part of A is used as a workspace. For example, if uplo is
     * CUBLAS_FILL_MODE_UPPER, upper triangle of A contains Cholesky factor U
     * and lower triangle of A is destroyed after potrfBatched.
     *
     * @param handle Handle to cuSolver context.
     * @param n The dimension of the symmetric matrices.
     * @param lda Leading dimension of matrix A (n).
     * @param fill The part of the dense matrix that is looked at and replaced.
     * @param info infoArray is an integer array of size batchsize. If
     * potrfBatched returns CUSOLVER_STATUS_INVALID_VALUE, infoArray[0] = -i
     * (less than zero), meaning that the i-th parameter is wrong (not counting
     * handle). If potrfBatched returns CUSOLVER_STATUS_SUCCESS but infoArray[i]
     * = k is positive, then i-th matrix is not positive definite and the
     * Cholesky factorization failed at row k.
     */
    public void choleskyFactorization(Handle handle, int n, int lda, IArray info, Fill fill) {
        checkNull(handle, info);
        checkPos(n, lda);

        int success = JCusolverDn.cusolverDnDpotrfBatched(
                handle.solverHandle(), // cuSolver handle
                fill.getFillMode(),
                n, // Matrix dimension
                pointer, // Input matrices (symmetric)
                lda, // Leading dimension                
                info.pointer, // Info array for errors
                length // Number of matrices
        );
        if (success != cusolverStatus.CUSOLVER_STATUS_SUCCESS)
            throw new RuntimeException("choleskyFactorization didn't work.  "
                    + "Here's the info array: " + info.toString());
    }

    /**
     * Performs batched LU factorization of general matrices.
     *https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-getrfbatched
     * This function computes the LU factorization of a sequence of general
     * matrices A. It factors each matrix A as A = P * L * U, where P is a
     * permutation matrix, L is lower triangular (or unit lower triangular in
     * the case of square matrices), and U is upper triangular.
     *
     * The factorization is stored in place. The pivot array contains the pivot
     * indices for each matrix.
     *
     * @param handle Handle to cuSolver context.
     * @param n The dimension of the matrices (nxn).
     * @param lda Leading dimension of matrix A.
     * @param pivotArray Array of pivots, where pivotArray[i] contains the pivot
     * indices for matrix i. This is an output array and whatever is in it will
     * be overwritten.
     * @param infoArray Info array for status/error reporting, where
     * infoArray[i] contains info for the i-th matrix. If infoArray[i] > 0, the
     * matrix is singular.
     */
    public void luFactorization(Handle handle, int n, int lda, IArray pivotArray, IArray infoArray) {
        checkNull(handle, pivotArray, infoArray);
        checkPos(n, lda);

        int success = JCublas2.cublasDgetrfBatched(
                handle.get(), // cuSolver handle
                n, // Matrix dimension
                pointer, // Input matrices (general)
                lda, // Leading dimension
                pivotArray.pointer, // Pivot indices for each matrix
                infoArray.pointer, // Info array for errors
                length // Number of matrices in the batch
        );
        if (success != cusolverStatus.CUSOLVER_STATUS_SUCCESS) {
            throw new RuntimeException("luFactorizationBatched didn't work. Here's the info array: " + infoArray.toString());
        }
    }

    /**
     * Solves a system of linear equations using the LU factorization obtained
     * by `getrfBatched`.
     *
     * This function solves a system of linear equations A * X = B for each
     * matrix A in the batch. It uses the LU decomposition from `getrfBatched`
     * to find the solution.
     *
     * @param handle Handle to cuBLAS context.
     * @param transposeA true if this matrix should be transposed. False
     * otherwise.
     * @param colsAndRowsA The dimension of the matrices (nxn).
     * @param colsB The number of right-hand sides, i.e., the number of columns
     * of the matrix B.
     * @param ldThis Leading dimension of matrix A.
     * @param ldb Leading dimension of matrix B.
     * @param pivotArray Pivot indices as generated by `getrfBatched`.
     * @param B The right-hand side matrices B. On exit, contains the solutions
     * X.
     * @param info Info array for status/error reporting. If infoArray[i] > 0,
     * matrix i is singular.
     */
    public void solveWithLUFactored(Handle handle, boolean transposeA, int colsAndRowsA, int colsB, int ldThis, int ldb, IArray pivotArray, DArray2d B, IArray info) {
        checkNull(handle, pivotArray, B, info);
        checkPos(colsAndRowsA, colsB, ldThis, ldb);

        int success = JCublas2.cublasDgetrsBatched(
                handle.get(), DArray.transpose(transposeA),
                colsAndRowsA, colsB,
                pointer, ldThis,
                pivotArray.pointer, // Pivot indices for each matrix
                B.pointer, ldb,
                info.pointer,
                length // Number of matrices in the batch
        );
        if (success != cublasStatus.CUBLAS_STATUS_SUCCESS) {
            throw new RuntimeException("solveWithLUFactorizationBatched didn't work. Here's the info array: " + info.toString());
        }
    }


    /* Doesn't work because Jacobiparms doesn't work.
     * 
     * Creates an auxiliary workspace for cusolverDnDsyevjBatched using
     * cusolverDnDsyevjBatched_bufferSize.
     *
     * @param handle The cusolverDn handle.
     * @param height The size of the matrices (nxn).
     * @param input The device pointer to the input matrices.
     * @param ldInput The leading dimension of the matrix A.
     * @param resultValues The device pointer to the eigenvalue array.
     * @param batchSize The number of matrices in the batch.
     * @param fill How is the matrix stored.
     * @param params The syevjInfo_t structure for additional parameters.
     * @return A Pointer array where the first element is the workspace size,
     * and the second element is the device pointer to the workspace.
     */
    public static int eigenWorkspaceSize(Handle handle,
            int height,
            DArray input,
            int ldInput,
            DArray resultValues,
            int batchSize,
            JacobiParams params,
            Fill fill) {
        int[] lwork = new int[1];

        JCusolverDn.cusolverDnDsyevjBatched_bufferSize(
                handle.solverHandle(),
                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR,
                fill.getFillMode(),
                height,
                input.pointer,
                ldInput,
                resultValues.pointer,
                lwork,
                params.getParams(),
                batchSize
        );

        return lwork[0];
    }

//Doesn't work because JacobiParams doesn't work.
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
     * The input matrices are replaced with the eigenvectors.
     *
     * Use createEigenWorkspace to calculate the size of the workspace.
     *
     * @param height The height and width of the matrices.
     * @param inputMatrices The input matrices, which must be stored in GPU
     * memory consecutively so that each matrix is column-major with leading
     * dimension lda, so the formula for random access is a_k[i, j] = A[i +
     * lda*j + lda*n*k]
     * @param ldInput The leading dimension of the input matrices.
     * @param resultValues Array to store the eigenvalues of the matrices.
     * @param workSpace An auxilery workspace.
     * @param batchCount The number of matrices in the batch.
     * @param cublasFillMode Fill mode for the symmetric matrix (upper or
     * lower).
     * @param jp a recourse needed by this method.
     * @param info An integer array. It's length should be batch count. It
     * stores error messages.
     */
    public static void computeEigen(Handle handle, int height,
            DArray inputMatrices, int ldInput, DArray resultValues,
            DArray workSpace, int batchCount, Fill cublasFillMode,
            JacobiParams jp, IArray info) {

        JCusolverDn.cusolverDnDsyevjBatched(
                handle.solverHandle(), // Handle to the cuSolver context
                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR, // Compute both eigenvalues and eigenvectors
                cublasFillMode.getFillMode(), // Indicates matrix is symmetric
                height, inputMatrices.pointer, ldInput,
                resultValues.pointer,
                workSpace.pointer, workSpace.length,
                info.pointer, // Array to store status info for each matrix
                jp.getParams(), // Jacobi algorithm parameters
                batchCount // Number of matrices in the batch
        );
//            // Step 5: Check for convergence status in info array
//            int[] infoHost = new int[batchCount]; // Host array to fetch status
//
//            try (Handle hand = new Handle()) {
//                info.get(hand, infoHost, 0, 0, batchCount);
//            }
//
//            for (int i = 0; i < batchCount; i++) {
//                if (infoHost[i] != 0) {
//                    System.err.println("Matrix " + i + " failed to converge: info = " + infoHost[i]);
//                }
//            }

    }

    /**
     * Solves a symmetric positive definite system of linear equations A * x =
     * b, where A is a symmetric matrix that has undergone Cholesky
     * factorization and B and X are matrices of right-hand side vectors and
     * solutions, respectively.
     *
     * This method utilizes the cuSolver library and the
     * `cusolverDnDpotrsBatched` function to solve a batch of systems using the
     * Cholesky factorization. The matrix A must be symmetric positive definite.
     *
     * The input matrix A is provided in packed format, with either the upper or
     * lower triangular part of the matrix being supplied based on the `fillA`
     * parameter.
     *
     * This method checks for valid inputs and initializes the info array if not
     * provided. The `info` array stores error messages for each matrix in the
     * batch.
     *
     * @param handle The cuSolver handle, which is required for cuSolver library
     * operations. Must not be null.
     * @param fillA Indicates whether the upper or lower triangular part of A is
     * stored. It should be either {@link Fill#UPPER} or {@link Fill#LOWER}.
     * @param heightA The number of rows of the matrix A.
     * @param lda The leading dimension of the matrix A.
     * @param b The right-hand side matrix B and will store the solution matrix
     * X after computation. The input matrix must be stored in column-major
     * order on the GPU.
     * @param ldb The leading dimension of the matrix B.
     * @param info An optional output array to store the status of each system
     * in the batch. If `info == null`, an array will be created internally. If
     * info is not null, it must have a length equal to the number of matrices
     * in the batch.
     *
     * @throws IllegalArgumentException if the handle, fillA, or b is null.
     * @throws IllegalArgumentException if any of the dimensions (heightA,
     * widthBAndX, lda, ldb) are not positive.
     */
    public void solveCholesky(Handle handle, Fill fillA, int heightA, int lda, DArray2d b, int ldb, IArray info) {

        checkNull(handle, fillA, b);
        checkPos(heightA, lda, ldb);

        boolean cleanInfo = false;
        if (info == null) {
            info = IArray.empty(length);
            cleanInfo = true;
        }
        JCusolverDn.cusolverDnDpotrsBatched(
                handle.solverHandle(), fillA.getFillMode(),
                heightA, 1,
                pointer, lda,
                b.pointer, ldb,
                info.pointer,
                length
        );
        if (cleanInfo) info.close();
    }

    @Override
    public String toString() {

        try (Handle hand = new Handle()) {
            Pointer[] pointers = emptyHostArray(length);

            get(hand, Pointer.to(pointers), 0, 0, length);

            hand.synch();
            return Arrays.toString(pointers);
        }
    }

    public static void main(String[] args) {
        try (Handle handle = new Handle()) {

            try (
                    DArray data = new DArray(handle, 1, 5, 10); DArray2d a2d = new DArray2d(handle, new DArray[]{data});) {

                handle.synch();

                System.out.println(a2d);
                System.out.println(a2d.get(handle, 0).toString());
            }

        }

    }

}
