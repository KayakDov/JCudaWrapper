package array;

import static array.Array.checkNull;
import static array.Array.checkPos;
import static array.DArray.cpuPointer;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.cusolverEigMode;
import resourceManagement.Handle;
import resourceManagement.MySyevjInfo;

/**
 * A class for a batch of consecutive arrays.
 *
 * @author E. Dov Neimand
 */
public class DBatchArray extends DArray {

//    TODO: implement cusolverDnSgesvdjBatched
    public final int stride;

    /**
     * @param p A pointer to the first element.
     * @param length The total number of elements, This should be a multiple of
     * strideSize.
     * @param strideSize The distance from the first element of one subsequence to the first element of the next.
     * @param sectionLength The length of each subsequence.
     */
    protected DBatchArray(CUdeviceptr p, int length, int strideSize, int sectionLength) {
        super(p, length);
        this.stride = strideSize;        
    }

    public int batchCount() {
        return length / stride;
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
    public int eigenWorkspaceSize(Handle handle,
            int height,           
            int ldInput,
            DArray resultValues,
            MySyevjInfo params,
            DPointerArray.Fill fill) {
        int[] lwork = new int[1];

        JCusolverDn.cusolverDnDsyevjBatched_bufferSize(
                handle.solverHandle(),
                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR,
                fill.getFillMode(),
                height,
                pointer,
                ldInput,
                resultValues.pointer,
                lwork,
                params.getParams(),
                batchCount()
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
     *
     * This, the input matrices, which must be stored in GPU memory
     * consecutively so that each matrix is column-major with leading dimension
     * lda, so the formula for random access is a_k[i, j] = A[i + lda*j +
     * lda*n*k]
     *
     * The input matrices are replaced with the eigenvectors.
     *
     * Use createEigenWorkspace to calculate the size of the workspace.
     *
     * @param handle
     * @param height The height and width of the matrices.
     *
     * @param ldInput The leading dimension of the input matrices.
     * @param resultValues Array to store the eigenvalues of the matrices.
     * @param workSpace An auxilery workspace.     
     * @param cublasFillMode Fill mode for the symmetric matrix (upper or
     * lower).
     * @param jp a recourse needed by this method.
     * @param info An integer array. It's length should be batch count. It
     * stores error messages.
     */
    public void computeEigen(Handle handle, int height,
            int ldInput, DArray resultValues,
            DArray workSpace, DPointerArray.Fill cublasFillMode,
            MySyevjInfo jp, IArray info) {

        JCusolverDn.cusolverDnDsyevjBatched(handle.solverHandle(), // Handle to the cuSolver context
                cusolverEigMode.CUSOLVER_EIG_MODE_VECTOR, // Compute both eigenvalues and eigenvectors
                cublasFillMode.getFillMode(), // Indicates matrix is symmetric
                height, pointer, ldInput,
                resultValues.pointer,
                workSpace.pointer, workSpace.length,
                info.pointer, // Array to store status info for each matrix
                jp.getParams(), // Jacobi algorithm parameters
                batchCount()
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
     * @param matB Pointer to the batched matrix B in GPU memory.
     * @param ldb Leading dimension of matrix B (the number of elements between
     * consecutive columns in memory).
     * @param timesResult Scalar multiplier applied to each result matrix before
     * adding the matrix-matrix product.
     * @param ldResult Leading dimension of the result matrix (the number of
     * elements between consecutive columns in memory).
     *
     */
    public void multMatMatStridedBatched(Handle handle, boolean transA, boolean transB,
            int aRows, int aColsBRows, int bCols, double timesAB, DBatchArray matA,
            int lda, DBatchArray matB, int ldb, double timesResult,
            int ldResult) {

        checkNull(handle, matA, matB);
        checkPos(aRows, bCols, ldb, ldResult);
        checkAgainstLength(aRows * bCols * batchCount() - 1);

        JCublas2.cublasDgemmStridedBatched(
                handle.get(),
                DArray.transpose(transA), DArray.transpose(transB),
                aRows, bCols, aColsBRows,
                cpuPointer(timesAB),
                matA.pointer, lda, matA.stride,
                matB.pointer, ldb, matB.stride,
                cpuPointer(timesResult), pointer, ldResult, stride,
                batchCount()
        );
    }
    
    /**
     * An empty batch array.
     * @param batchSize The number of subsequences.
     * @param strideSize The size of each subsequence.
     * @return An empty batch array.
     */
    public static DBatchArray empty(int batchSize, int strideSize){
        int size = strideSize * batchSize;
        return new DBatchArray(
                Array.empty(size, PrimitiveType.DOUBLE), 
                size, 
                strideSize, 0
        );
    }
    
    /**
     *
     * @param handle
     * @return An array of pointers to each of the subsequences.
     */
    public DPointerArray getPointerArray(Handle handle){
        return super.getPointerArray(handle, stride);
    }
    
//    TODO: implement cusolverDnSgesvdjBatched
}
