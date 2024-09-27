package storage;

import processSupport.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasDiagType;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasGemmAlgo;
import jcuda.jcublas.cublasOperation;

/**
 * This class provides functionalities to create and manipulate double arrays on
 * the GPU.
 *
 * For more methods that might be useful here, see:
 * https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
 *
 * TODO: create arrays other than double.
 *
 * @author E. Dov Neimand
 */
public class DArray extends Array {

    /**
     * Creates a GPU array from a CPU array.
     *
     * @param handle The gpu handle that manages this operation.
     * @param values The array to be copied to the GPU.
     * @throws IllegalArgumentException if the values array is null.
     */
    public DArray(Handle handle, double... values) {
        this(Array.empty(values.length, PrimitiveType.DOUBLE), values.length);
        copy(handle, this, values, 0, 0, values.length);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray copy(Handle handle) {
        DArray copy = DArray.empty(length);
        get(handle, copy, 0, 0, length);
        return copy;
    }

    /**
     * Constructs an array with a given GPU pointer and length.
     *
     * @param p A pointer to the first element of the array on the GPU.
     * @param length The length of the array.
     */
    protected DArray(CUdeviceptr p, int length) {
        super(p, length, PrimitiveType.DOUBLE);
    }

    /**
     * Creates an empty DArray with the specified size.
     *
     * @param size The number of elements in the array.
     * @return A new DArray with the specified size.
     * @throws ArrayIndexOutOfBoundsException if size is negative.
     */
    public static DArray empty(int size) {
        checkPos(size);
        return new DArray(Array.empty(size, PrimitiveType.DOUBLE), size);
    }

    /**
     * Copies contents from a CPU array to a GPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param to The destination GPU array.
     * @param fromArray The source CPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param length The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public static void copy(Handle handle, DArray to, double[] fromArray, int toIndex, int fromIndex, int length) {
        checkNull(fromArray, to);

        Array.copy(
                handle,
                to,
                Pointer.to(fromArray),
                toIndex,
                fromIndex,
                length,
                PrimitiveType.DOUBLE
        );
    }

    /**
     * A pointer to a singleton array containing d.
     *
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    static Pointer cpuPointer(double d) {
        return Pointer.to(new double[]{d});
    }

    /**
     * Copies the contents of this GPU array to a CPU array.
     *
     * @param to The destination CPU array.
     * @param toStart The index in the destination array to start copying to.
     * @param fromStart The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void get(Handle handle, double[] to, int toStart, int fromStart, int length) {
        checkNull(to);
        get(handle, Pointer.to(to), toStart, fromStart, length);
    }

    /**
     * Exports a portion of this GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param fromStart The starting index in this GPU array.
     * @param length The number of elements to export.
     * @return A CPU array containing the exported portion.
     * @throws IllegalArgumentException if fromStart or length is out of bounds.
     */
    public double[] get(Handle handle, int fromStart, int length) {
        double[] export = new double[length];
        get(handle, export, 0, fromStart, length);
        return export;
    }

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @return A CPU array containing all elements of this GPU array.
     */
    public double[] get(Handle handle) {
        return get(handle, 0, length);
    }

    /**
     * Copies from this vector to another with increments.
     *
     * @param handle handle to the cuBLAS library context.
     * @param to The array to copy to.
     * @param toStart The index to start copying to.
     * @param toInc stride between consecutive elements of the array copied to.
     * @param fromStart The index to start copying from.
     * @param fromInc stride between consecutive elements of this array.
     * @param length The number of elements to copy.
     */
    public void get(Handle handle, DArray to, int toStart, int fromStart, int toInc, int fromInc, int length) {

        JCublas2.cublasDcopy(handle.get(),
                length,
                pointer(fromStart),
                fromInc,
                to.pointer(toStart),
                toInc
        );
    }

    /**
     * Copies from to vector from another with increments.
     *
     * @param handle handle to the cuBLAS library context.
     * @param from The array to copy from.
     * @param fromStart The index to begin copying from.
     * @param toInc stride between consecutive elements of the array copied to.
     * @param toStart The index to begin copying to.
     * @param fromInc stride between consecutive elements of this array.
     * @param length The number of elements to copy.
     */
    public void set(Handle handle, DArray from, int toStart, int fromStart, int toInc, int fromInc, int length) {

        from.get(handle, this, toStart, fromStart, toInc, fromInc, length);
    }

    /**
     * Copies a CPU array to this GPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param from The source CPU array.
     * @param toIndex The index in this GPU array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param size The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or size is
     * negative.
     */
    public void set(Handle handle, double[] from, int toIndex, int fromIndex, int size) {
        copy(handle, this, from, toIndex, fromIndex, size);
    }

    /**
     * Copies a CPU array to this GPU array.
     *
     * @param handle handle to the cuBLAS library context.
     * @param from The source CPU array.
     * @throws IllegalArgumentException if from is null.
     */
    public final void set(Handle handle, double[] from) {
        set(handle, from, 0, 0, from.length);
    }

    /**
     * Copies a CPU array to this GPU array starting from a specified index.
     *
     * @param handle The handle.
     * @param from The source CPU array.
     * @param toIndex The index in this GPU array to start copying to.
     * @throws IllegalArgumentException if from is null.
     */
    public void set(Handle handle, double[] from, int toIndex) {
        set(handle, from, toIndex, 0, from.length);
    }

    /**
     * A sub array of this array. Note, this is not a copy and changes to this
     * array will affect the sub array and vice versa.
     *
     * @param start The beginning of the sub array.
     * @param length The length of the sub array.
     * @return A sub Array.
     */
    public DArray subArray(int start, int length) {
        checkPos(start, length);
        return new DArray(pointer(start), length);
    }

    /**
     * A sub array of this array. Note, this is not a copy and changes to this
     * array will affect the sub array and vice versa. The length of the new
     * array will go to the end of this array.
     *
     * @param start The beginning of the sub array.
     * @return A sub array.
     */
    public DArray subArray(int start) {
        checkPos(start);
        return new DArray(pointer(start), length - start);
    }

    /**
     * Sets the value at the given index.
     *
     * @param handle handle to the cuBLAS library context.
     * @param index The index the new value is to be assigned to.
     * @param val The new value at the given index.
     */
    public void set(Handle handle, int index, double val) {
        checkPos(index);
        checkAgainstLength(index);
        set(handle, new double[]{val}, index);
    }

    /**
     * Gets the value from the given index.
     *
     * @param index The index the value is to be retrieved from.
     * @return The value at index.
     */
    public DSingleton get(int index) {
        checkPos(index);
        checkAgainstLength(index);
        return new DSingleton(this, index);
    }

    /**
     * Computes a matrix-matrix addition (GEAM) or transpose with
     * double-precision.
     *
     * This function computes this = alpha * op(A) + beta * op(B), where op(X)
     * can be X or X^T (the transpose of X). For matrix transposition, set op(A)
     * to A^T (transpose), and set B as null with beta = 0.
     *
     * @param handle The CUBLAS context (a pointer to the initialized
     * cublasHandle_t).
     * @param transA Operation type for matrix A. Can be one of: CUBLAS_OP_N (no
     * transpose), CUBLAS_OP_T (transpose), CUBLAS_OP_C (conjugate transpose).
     * @param transB Operation type for matrix B. Can be one of: CUBLAS_OP_N (no
     * transpose), CUBLAS_OP_T (transpose), CUBLAS_OP_C (conjugate transpose).
     * For transpose operation, set transB to CUBLAS_OP_N and B as null.
     * @param heightA The number of rows of the matrix A (before transposition).
     * @param widthA The number of columns of the matrix A (before
     * transposition).
     * @param alpha Pointer to the scalar alpha (usually 1.0 for transposition).
     * @param a Pointer to the input matrix A on the GPU (before transposition).
     * @param lda The number of elements between the first element of each
     * column, this is usually height, but can be more if the matrix described
     * is a submatrix.
     * @param beta Pointer to the scalar beta (set to 0.0 for transposition).
     * @param b Pointer to the matrix B on the GPU (can be null for
     * transposition). 0).
     * @param ldb This should be 0 if B is null.
     * @param ldc ldc: Leading dimension of this matrix (ldc ≥ max(1, n) after
     * transposition).
     *
     * @return Status code from CUBLAS library: CUBLAS_STATUS_SUCCESS if the
     * operation was successful, or an appropriate error code otherwise.
     *
     */
    public int matrixAddWithTranspose(
            Handle handle,
            boolean transA,
            boolean transB,
            int heightA,
            int widthA,
            double alpha,
            DArray a,
            int lda,
            double beta,
            DArray b,
            int ldb,
            int ldc
    ) {
        checkNull(handle, a, b);
        checkPos(heightA, widthA);
        checkLowerBound(heightA, lda, ldb, ldc);
        a.checkAgainstLength(heightA * widthA - 1);
        b.checkAgainstLength(heightA * widthA - 1);
        checkAgainstLength(heightA * widthA - 1);

        return JCublas2.cublasDgeam(
                handle.get(),
                transA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                transB ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                heightA, widthA, cpuPointer(alpha), a.pointer, lda,
                cpuPointer(beta),
                b == null ? null : b.pointer,
                ldb, pointer, ldc);
    }

    /**
     * Performs the rank-1 update: This is outer product.
     *
     * <pre>
     * this = multProd * X * Y^T + this
     * </pre>
     *
     * Where X is a column vector and Y^T is a row vector.
     *
     * @param handle handle to the cuBLAS library context.
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param multProd Scalar applied to the outer product of X and Y^T.
     * @param vecX Pointer to vector X in GPU memory.
     * @param incX When iterating thought the elements of x, the jump size. To
     * read all of x, set to 1.
     * @param vecY Pointer to vector Y in GPU memory.
     * @param incY When iterating though the elements of y, the jump size.
     *
     */
    public void outerProd(Handle handle, int rows, int cols, double multProd, DArray vecX, int incX, DArray vecY, int incY, int lda) {
        checkNull(handle, vecX, vecY);
        checkPos(lda, cols);
        checkLowerBound(1, incY, incX);
        checkAgainstLength(lda * cols - 1);

        JCublas2.cublasDger(handle.get(), rows, cols, cpuPointer(multProd), vecX.pointer, incX, vecY.pointer, incY, pointer, lda);
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm): This method
     * synchronizes the handle.
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars that will be squared.
     * @param inc The stride step over this array.
     * @return The Euclidean norm of this vector.
     */
    public double norm(Handle handle, int length, int inc) {
        double[] result = new double[1];
        norm(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm):
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars that will be squared.
     * @param inc The stride step over this array.
     * @param result where the result is to be stored.
     * @param toIndex The index in the result array to store the result;
     */
    public void norm(Handle handle, int length, int inc, double[] result, int toIndex) {
        //Warning, the cuda documentation says that the return pointer can 
        //either device or host, but there seems to be a core dump whenever
        //I try to put the result on the host.
        checkNull(handle, result);
        JCublas2.cublasDnrm2(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.DOUBLE));
    }

    /**
     * Finds the index of the element with the minimum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @param result where the result (index of the minimum absolute value) is
     * to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void argMinAbs(Handle handle, int length, int inc, int[] result, int toIndex) {
        checkNull(handle, result);
        JCublas2.cublasIdamin(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.INT));
        result[toIndex] -= 1;//It looks like the cuda methods are index-1 based.
    }

    /**
     * Finds the index of the element with the maximum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @param result where the result (index of the maximum absolute value) is
     * to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void argMaxAbs(Handle handle, int length, int inc, int[] result, int toIndex) {
        checkNull(handle, result);
        JCublas2.cublasIdamax(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.INT));
        result[toIndex] -= 1; //It looks like the cuda methods are index-1 based.
    }

    /**
     * Finds the index of the element with the minimum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of min(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @return The index of the lement with the minimum absolute value.
     */
    public int argMinAbs(Handle handle, int length, int inc) {
        int[] result = new int[1];
        argMinAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Finds the index of the element with the maximum absolute value in the
     * vector X:
     *
     * <pre>
     * result = index of max(|X[0]|, |X[1]|, ..., |X[n-1]|)
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of elements to search.
     * @param inc The stride step over the array.
     * @return The index of the element with greatest absolute value.
     *
     */
    public int argMaxAbs(Handle handle, int length, int inc) {
        int[] result = new int[1];
        argMaxAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the sum of the absolute values of the vector X (1-norm):
     *
     * <pre>
     * result = |X[0]| + |X[1]| + ... + |X[n-1]|
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars to include in the sum.
     * @param inc The stride step over the array.
     * @return The l1 norm of the vector.
     */
    public double sumAbs(Handle handle, int length, int inc) {
        double[] result = new double[1];
        DArray.this.sumAbs(handle, length, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the sum of the absolute values of the vector X (1-norm):
     *
     * <pre>
     * result = |X[0]| + |X[1]| + ... + |X[n-1]|
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param length The number of scalars to include in the sum.
     * @param inc The stride step over the array.
     * @param result where the result is to be stored.
     * @param toIndex The index in the result array to store the result.
     */
    public void sumAbs(Handle handle, int length, int inc, double[] result, int toIndex) {
        checkNull(handle, result);
        JCublas2.cublasDasum(handle.get(), length, pointer, inc, Pointer.to(result).withByteOffset(toIndex * Sizeof.DOUBLE));
    }

    /**
     * Performs the matrix-vector multiplication:
     *
     * <pre>
     * this = timesAx * op(A) * X + beta * this
     * </pre>
     *
     * Where op(A) can be A or its transpose.
     *
     * @param handle handle to the cuBLAS library context.
     * @param transA Specifies whether matrix A is transposed (true for
     * transpose and false for not.)
     * @param aRows The number of rows in matrix A.
     * @param aCols The number of columns in matrix A.
     * @param timesAx Scalar multiplier applied to the matrix-vector product.
     * @param matA Pointer to matrix A in GPU memory.
     * @param lda The distance between the first element of each column of A.
     * @param vecX Pointer to vector X in GPU memory.
     * @param incX The increments taken when iterating over elements of X. This
     * is usually1 1. If you set it to 2 then you'll be looking at half the
     * elements of x.
     * @param beta Scalar multiplier applied to vector Y before adding the
     * matrix-vector product.
     * @param inc the increment taken when iterating over elements of this
     * array.
     * @return this array after this = timesAx * op(A) * X + beta*this
     */
    public DArray multMatVec(Handle handle, boolean transA, int aRows, int aCols, double timesAx, DArray matA, int lda, DArray vecX, int incX, double beta, int inc) {
        checkNull(handle, matA, vecX);
        checkPos(aRows, aCols);
        checkLowerBound(1, inc, incX);
        matA.checkAgainstLength(aRows * aCols);

        JCublas2.cublasDgemv(
                handle.get(),
                transA ? 'T' : 'N',
                aRows,
                aCols,
                cpuPointer(timesAx),
                matA.pointer,
                lda,
                vecX.pointer,
                incX,
                cpuPointer(beta),
                pointer,
                inc
        );
        return this;
    }

    /**
     * Multiplies this vector by a banded matrix and adds the result to the
     * vector.
     *
     * this = timesAx * op(A) * X + timesThis * this
     *
     * A banded matrix is a sparse matrix where the non-zero elements are
     * confined to a diagonal band, comprising the main diagonal, a fixed number
     * of subdiagonals below the main diagonal, and a fixed number of
     * superdiagonals above the main diagonal. Banded matrices are often used to
     * save space, as only the elements within the band are stored, and the rest
     * are implicitly zero.
     *
     * In this method, the banded matrix is represented by the {@link DArray} M,
     * and the structure of M is defined by the number of subdiagonals and
     * superdiagonals. The matrix is stored in a column-major order, with each
     * column being a segment of the band. The parameter `lda` indicates the
     * leading dimension of the banded matrix, which corresponds to the number
     * of rows in the compacted matrix representation. The elements of the band
     * are stored contiguously in memory, with zero-padding where necessary to
     * fill out the bandwidth of the matrix.
     *
     * Let M represent the column-major matrix that stores the elements of A in
     * a {@link DArray}. The first row of M corresponds to the top-rightmost
     * non-zero diagonal of A (the highest superdiagonal). The second row
     * corresponds to the diagonal that is one position below/left of the first
     * row, and so on, proceeding down the diagonals. The final row of M
     * contains the bottom-leftmost diagonal of A (the lowest subdiagonal).
     * Diagonals that do not fully extend across A are padded with zeros in M.
     * An element in A has the same column in M as it does in A.
     *
     * This method performs the matrix-vector multiplication between the banded
     * matrix A and the vector x using the JCublas `cublasDgbmv` function, which
     * supports operations on banded matrices. The result is scaled by `timesA`
     * and added to this vector scaled by `timesThis`.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param transposeA Whether to transpose matrix A before multiplying.
     * @param rowsA The number of rows in matrix A.
     * @param colsA The number of columns in matrix A.
     * @param subDiagonalsA The number of subdiagonals in matrix A.
     * @param superDiagonalA The number of superdiagonals in matrix A.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray} representing the banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted banded matrix representation.
     * @param x The {@link DArray} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray multBandMatVec(Handle handle, boolean transposeA, int rowsA, int colsA, int subDiagonalsA, int superDiagonalA, double timesA, DArray Ma, int ldm, DArray x, int incX, double timesThis, int inc) {
        JCublas2.cublasDgbmv(
                handle.get(),
                transposeA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                rowsA,
                colsA,
                subDiagonalsA,
                superDiagonalA,
                cpuPointer(timesA),
                Ma.pointer,
                ldm,
                x.pointer,
                incX,
                cpuPointer(timesThis),
                pointer,
                inc);

        return this;
    }

    /**
     * Solves the system of linear equations Ax = b, where A is a triangular
     * banded matrix, and x is the solution vector.
     *
     * b is this when the algorithm begins, and x is this when the algorithm
     * ends. That is to say the solution to Ax = this is stored in this.
     *
     * A triangular banded matrix is a special type of sparse matrix where the
     * non-zero elements are confined to a diagonal band around the main
     * diagonal and only the elements above (or below) the diagonal are stored,
     * depending on whether the matrix is upper or lower triangular.
     *
     * The matrix is stored in a compact banded format, where only the diagonals
     * of interest are represented to save space. For a lower diagonal matrix,
     * the first row represents the main diagonal, and subsequent rows represent
     * the diagonals progressively further from the main diagonal. An upper
     * diagonal matrix is stored with the last row as the main diagonal and the
     * first row as the furthest diagonal from the main.
     *
     * This method uses JCublas `cublasDtbsv` function to solve the system of
     * equations for the vector x.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param isUpper Indicates whether the matrix A is upper or lower
     * triangular. Use {@code cublasFillMode.CUBLAS_FILL_MODE_UPPER} for upper
     * triangular, or {@code cublasFillMode.CUBLAS_FILL_MODE_LOWER} for lower
     * triangular.
     * @param transposeA Whether to transpose the matrix A before solving the
     * system. Use {@code cublasOperation.CUBLAS_OP_T} for transpose, or
     * {@code cublasOperation.CUBLAS_OP_N} for no transpose.
     * @param onesOnDiagonal Specifies whether the matrix A is unit triangular
     * ({@code cublasDiagType.CUBLAS_DIAG_UNIT}) or non-unit triangular
     * ({@code cublasDiagType.CUBLAS_DIAG_NON_UNIT}).
     * @param rowsA The number of rows/columns of the matrix A (the order of the
     * matrix).
     * @param nonPrimaryDiagonals The number of subdiagonals or superdiagonals
     * in the triangular banded matrix.
     * @param Ma A compact form {@link DArray} representing the triangular
     * banded matrix A.
     * @param ldm The leading dimension of the banded matrix, which defines the
     * row count in the compacted matrix representation.
     *
     * @param inc The stride for stepping through the elements of b.
     * @return The updated {@link DArray} (b), now containing the solution
     * vector x.
     */
    public DArray solveTriangularBandedSystem(Handle handle, boolean isUpper, boolean transposeA, boolean onesOnDiagonal, int rowsA, int nonPrimaryDiagonals, DArray Ma, int ldm, int inc) {
        // Call the cublasDtbsv function to solve the system
        JCublas2.cublasDtbsv(
                handle.get(),
                isUpper ? cublasFillMode.CUBLAS_FILL_MODE_UPPER : cublasFillMode.CUBLAS_FILL_MODE_LOWER, // Upper or lower triangular matrix
                transposeA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N, // Whether to transpose A
                onesOnDiagonal ? cublasDiagType.CUBLAS_DIAG_UNIT : cublasDiagType.CUBLAS_DIAG_NON_UNIT, // Whether A is unit or non-unit triangular
                rowsA, // Number of rows/columns in A
                nonPrimaryDiagonals, // Number of subdiagonals/superdiagonals
                Ma.pointer, // Pointer to the compact form of matrix A
                ldm, // Leading dimension of Ma
                pointer, // Pointer to the right-hand side vector (b)
                inc);          // Stride through the elements of b

        return this;  // The result (solution vector x) is stored in b
    }

    @Override
    public String toString() {
        try (Handle handle = new Handle()) {
            return Arrays.toString(get(handle));
        }

    }

    /**
     * Multiplies this vector by a symmetric banded matrix and adds the result
     * to the vector.
     *
     * this = timesA * A * x + timesThis * this
     *
     * A symmetric banded matrix is a matrix where the non-zero elements are
     * confined to a diagonal band around the main diagonal, and the matrix is
     * symmetric (i.e., A[i][j] = A[j][i]). In a symmetric banded matrix, only
     * the elements within the band are stored, as the symmetry allows the upper
     * or lower part to be inferred. This storage technique reduces memory
     * usage.
     *
     * In this method, the symmetric banded matrix is represented by the A
     * stored in Ma, where only the upper (or lower) part of the matrix is
     * stored. The matrix is stored in a compact form, with each column being a
     * segment of the band. The parameter `ldm` indicates the leading dimension
     * of the matrix, which corresponds to the number of rows in the compacted
     * matrix representation. Only the non-zero diagonals of the matrix are
     * stored contiguously in memory.
     *
     * Let M represent the column-major matrix that stores the elements of the
     * symmetric banded matrix A in a {@link DArray}. The first row of M
     * corresponds to the main diagonal of A, and the subsequent rows correspond
     * to diagonals above or below the main diagonal. For instance, the second
     * row corresponds to the diagonal directly above the main diagonal, and so
     * on.
     *
     * This method performs the matrix-vector multiplication between the
     * symmetric banded matrix A and the vector x using the JCublas
     * `cublasDsbmv` function, which supports operations on symmetric banded
     * matrices. The result is scaled by `timesA` and added to this vector
     * scaled by `timesThis`.
     *
     * @param handle The JCublas handle required for GPU operations.
     * @param upper Whether the upper triangular part of the matrix is stored.
     * @param colA The order of the symmetric matrix A (number of rows and
     * columns).
     * @param diagonals The number of subdiagonals or superdiagonals in the
     * matrix.
     * @param timesA Scalar multiplier for the matrix-vector product.
     * @param Ma A compact form {@link DArray} representing the symmetric banded
     * matrix A.
     * @param ldm The leading dimension of the matrix, defining the row count in
     * the compacted matrix representation.
     * @param x The {@link DArray} representing the input vector to be
     * multiplied.
     * @param incX The stride for stepping through the elements of x.
     * @param timesThis Scalar multiplier for this vector, to which the result
     * of the matrix-vector multiplication is added.
     * @param inc The stride for stepping through the elements of this vector.
     * @return The updated vector (this), after the matrix-vector multiplication
     * and addition.
     */
    public DArray multSymBandMatVec(Handle handle, boolean upper, int colA, int diagonals, double timesA, DArray Ma, int ldm, DArray x, int incX, double timesThis, int inc) {
        JCublas2.cublasDsbmv(
                handle.get(),
                upper ? cublasFillMode.CUBLAS_FILL_MODE_UPPER : cublasFillMode.CUBLAS_FILL_MODE_LOWER,
                colA,
                diagonals,
                cpuPointer(timesA),
                Ma.pointer,
                ldm,
                x.pointer,
                incX,
                cpuPointer(timesThis),
                pointer,
                inc);

        return this;
    }

    /**
     * Fills a matrix with a scalar value directly on the GPU using a CUDA
     * kernel.
     *
     * This function sets all elements of the matrix A to the given scalar
     * value. The matrix A is stored in column-major order, and the leading
     * dimension of A is specified by lda.
     *
     * In contrast to the method that doesn't use a handle, this one
     *
     * @param handle A handle.
     * @param fill the scalar value to set all elements of A
     * @param inc The increment with which the method iterates over the array.
     * @return this;
     */
    public DArray fill(Handle handle, double fill, int inc) {
        checkPos(inc);
        checkNull(handle);

        DSingleton from = new DSingleton(handle, fill);
        set(handle, from, 0, 0, inc, 0, Math.ceilDiv(length, inc));
        return this;
    }

    /**
     * Fills a matrix with a value.
     *
     * @param handle handle to the cuBLAS library context.
     * @param height The height of the matrix.
     * @param width The width of the matrix.
     * @param lda The distance between the first element of each column of the
     * matrix. This should be at least the height of the matrix.
     * @param fill The value the matrix is to be filled with.
     * @return this, after having been filled.
     */
    public DArray fillMatrix(Handle handle, int height, int width, int lda, double fill) {
        checkPos(height, width);
        checkLowerBound(height, lda);
        checkAgainstLength(height * width - 1);

        if (height == lda) {
            if (fill == 0) {
                subArray(0, width * height).fill0(handle);
                return this;
            }
            return subArray(0, width * height).fill(handle, fill, 1);
        }

        try (DArray filler = DArray.empty(width * height).fill(handle, fill, 1)) {

            addAndSet(handle, false, false, height, width, 1, filler, height, 0, this, lda, lda);
        }

        return this;
    }

    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * This method synchronizes the handle.
     *
     * @param handle handle to the cuBLAS library context.
     * @param incX The number of spaces to jump when incrementing forward
     * through x.
     * @param inc The number of spaces to jump when incrementing forward through
     * this array.
     * @param x Pointer to vector X in GPU memory.
     * @return The dot product of X and Y.
     */
    public double dot(Handle handle, DArray x, int incX, int inc) {
        double[] result = new double[1];
        dot(handle, x, incX, inc, result, 0);
        handle.synch();
        return result[0];
    }

    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param incX The number of spaces to jump when incrementing forward
     * through x.
     * @param inc The number of spaces to jump when incrementing forward through
     * this array.
     * @param x Pointer to vector X in GPU memory.
     * @param result The array the answer should be put in.
     * @param resultInd The index of the array the answer should be put in.
     */
    public void dot(Handle handle, DArray x, int incX, int inc, double[] result, int resultInd) {
        checkNull(handle, x, result);
        checkPos(resultInd, inc, incX);
        JCublas2.cublasDdot(handle.get(), length, x.pointer, incX, pointer, inc, Pointer.to(result).withByteOffset(resultInd * Sizeof.DOUBLE));

    }

    /**
     * Performs the matrix-matrix multiplication using double precision (Dgemm)
     * on the GPU:
     *
     * <pre>
     * this = op(A) * op(B) + this
     * </pre>
     *
     * Where op(A) and op(B) represent A and B or their transposes based on
     * `transa` and `transb`.
     *
     * @param handle There should be one handle in each thread.
     * @param transposeA True opA should be transpose, false otherwise.
     * @param transposeB True if opB should be transpose, false otherwise.
     * @param aRows The number of rows of matrix C and matrix A (if
     * !transposeA).
     * @param bCols The number of columns of this matrix and matrix B (if
     * !transposeP).
     * @param aCols The number of columns of matrix A (if !transposeA) and rows
     * of matrix B (if !transposeB).
     * @param timesAB A scalar to be multiplied by AB.
     * @param a Pointer to matrix A, stored in GPU memory. successive rows in
     * memory, usually equal to ARows).
     * @param lda The number of elements between the first element of each
     * column of A. If A is not a subset of a larger data set, then this will be
     * the height of A.
     * @param b Pointer to matrix B, stored in GPU memory.
     * @param ldb @see lda
     * @param timesCurrent This is multiplied by the current array first and
     * foremost. Set to 0 if the current array is meant to be empty, and set to
     * 1 to add the product to the current array as is.
     * @param ldc @see ldb
     */
    public void multMatMat(Handle handle, boolean transposeA, boolean transposeB, int aRows,
            int bCols, int aCols, double timesAB, DArray a, int lda, DArray b, int ldb, double timesCurrent, int ldc) {
        checkNull(handle, a, b);
        checkPos(aRows, bCols, aCols, lda, ldb, ldc);
        checkLowerBound(aRows, lda);
        checkLowerBound(aCols, ldb);
        checkLowerBound(aRows, ldc);
        a.checkAgainstLength(aCols * lda - 1);
        b.checkAgainstLength(bCols * ldb - 1);
        checkAgainstLength(aRows * bCols - 1);

        // Check the transpose options and set the corresponding CUBLAS operations.
        int transA = transposeA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
        int transB = transposeB ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;

        // Perform matrix multiplication using JCublas2
        JCublas2.cublasDgemm(
                handle.get(), // cublas handle
                transA, // Transpose operation for A
                transB, // Transpose operation for B
                aRows, // Number of rows in matrix A
                bCols, // Number of columns in matrix B
                aCols, // Number of columns in matrix A
                cpuPointer(timesAB), // Scalar to multiply with A*B
                a.pointer, // Matrix A (GPU pointer)
                lda, // Leading dimension of A
                b.pointer, // Matrix B (GPU pointer)
                ldb, // Leading dimension of B
                cpuPointer(timesCurrent), // Scalar to multiply with the current matrix in DArray (this)
                pointer, // Matrix C (the result matrix, stored in this DArray)
                ldc // Leading dimension of C
        );
    }

    /**
     * Performs the vector addition:
     *
     * <pre>
     * this = timesX * X + this
     * </pre>
     *
     * This operation scales vector X by alpha and adds it to vector Y.
     *
     * @param handle handle to the cuBLAS library context.
     * @param timesX Scalar used to scale vector X.
     * @param x Pointer to vector X in GPU memory.
     * @param incX The number of elements to jump when iterating forward through
     * x.
     * @param inc The number of elements to jump when iterating forward through
     * this.
     * @return this
     */
    public DArray addToMe(Handle handle, double timesX, DArray x, int incX, int inc) {
        checkNull(handle, x);
        checkLowerBound(1, inc);

        JCublas2.cublasDaxpy(handle.get(), length, Pointer.to(new double[]{timesX}), x.pointer, incX, pointer, inc);
        return this;
    }

    /**
     * Performs matrix addition or subtraction.
     *
     * <p>
     * This function computes C = alpha * A + beta * B, where A, B, and C are
     * matrices.
     * </p>
     *
     * @param handle the cuBLAS context handle
     * @param transA specifies whether matrix A is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param transB specifies whether matrix B is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param height number of rows of matrix C
     * @param width number of columns of matrix C
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param lda leading dimension of matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @param ldb leading dimension of matrix B
     * @param ldc leading dimension of matrix C
     * @return this
     *
     */
    public DArray addAndSet(Handle handle, boolean transA, boolean transB, int height,
            int width, double alpha, DArray a, int lda, double beta, DArray b,
            int ldb, int ldc) {
        checkNull(handle, a, b);
        checkPos(height, width);
        checkAgainstLength(height * width - 1);

        JCublas2.cublasDgeam(
                handle.get(),
                transA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                transB ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                height,
                width,
                cpuPointer(alpha),
                a.pointer,
                lda,
                cpuPointer(beta),
                b.pointer,
                ldb,
                pointer,
                ldc);

        return this;
    }

    /**
     * Scales this vector by the scalar mult:
     *
     * <pre>
     * this = mult * this
     * </pre>
     *
     * @param handle handle to the cuBLAS library context.
     * @param mult Scalar multiplier applied to vector X.
     * @param inc The number of elements to jump when iterating forward through
     * this array.
     * @return this;
     *
     *
     */
    public DArray multMe(Handle handle, double mult, int inc) {
        checkNull(handle);
        checkLowerBound(1, inc);
        JCublas2.cublasDscal(handle.get(), Math.ceilDiv(length, inc), Pointer.to(new double[]{mult}), pointer, inc);
        return this;
    }

    /**
     * TODO: put a version of this in the matrix class.
     *
     * Performs symmetric batch matrix-matrix multiplication using
     * cublasDsyrkBatched.
     *
     * Computes this = A * A^T + timesThis * this for each matrix in the batch,
     * ensuring C is symmetric.
     *
     * @param handle CUBLAS handle for managing the operation.
     * @param transpose
     * @param uplo Specifies which part of the matrix is being used (upper or
     * lower).
     * @param resultRowsCols The number of rows/columns of the result matrices.
     * @param cols The number of columns of A (for C = A * A^T).
     * @param alpha Scalar multiplier for A * A^T.
     * @param a Pointer array to the input matrices.
     * @param lda Leading dimension of A.
     * @param timesThis Scalar multiplier for the existing C matrix (usually 0
     * for new computation).
     * @param ldThis Leading dimension of C.
     *
     */
    public void matrixSquared(
            Handle handle,
            boolean transpose,
            int uplo, // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
            int resultRowsCols,
            int cols,
            double alpha,
            DArray a,
            int lda,
            double timesThis,
            int ldThis) {

        JCublas2.cublasDsyrk(
                handle.get(),
                uplo,
                transpose ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T, // Normal operation (no transpose)
                resultRowsCols,
                cols,
                cpuPointer(alpha),
                a.pointer,
                lda,
                cpuPointer(alpha),
                pointer,
                ldThis
        );
    }

}

//    //TODO: use cuSolver for sovling equations and eigan values and vectors
