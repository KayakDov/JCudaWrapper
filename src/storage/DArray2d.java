package storage;

import java.util.Arrays;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.cudaDataType;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasGemmAlgo;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.apache.commons.math3.exception.DimensionMismatchException;
import processSupport.Handle;

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
                transA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N, // Operation on A (transpose or not)
                transB ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N, // Operation on B (transpose or not)
                aRows, bCols, aColsBRows, // Number of columns of A / rows of B
                cpuPointer(timesAB),
                A.pointer, lda, // Leading dimension of A
                B.pointer, ldb, // Leading dimension of B
                cpuPointer(timesResult), pointer, ldResult, // Leading dimension of result matrices
                batchCount // Number of matrices to multiply
        );
    }

}
