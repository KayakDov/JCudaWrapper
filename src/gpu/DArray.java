package gpu;

import java.util.Arrays;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;

/**
 * This class provides functionalities to create and manipulate double arrays on
 * the GPU.
 *
 * TODO: create arrays other than double.
 * 
 * @author E. Dov Neimand
 */
public class DArray extends Array {

    /**
     * Creates a GPU array from a CPU array.
     *
     * @param values The array to be copied to the GPU.
     * @throws IllegalArgumentException if the values array is null.
     */
    public DArray(double... values) {
        this(empty(values.length), values.length);
        copy(this, values, 0, 0, values.length);
    }

    /**
     * Creates a copy of this array.
     *
     * @return A new DArray that is a copy of this array.
     */
    @Override
    public DArray copy() {
        DArray copy = DArray.emptyArray(length);
        get(copy, 0, 0, length);
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
    public static DArray emptyArray(int size) {
        checkPos(size);
        return new DArray(empty(size), size);
    }

    /**
     * Allocates space on the GPU and returns a pointer to the allocated space.
     *
     * @param size The number of elements to allocate space for.
     * @return A pointer to the allocated space.
     * @throws ArrayIndexOutOfBoundsException if size is negative.
     */
    protected static CUdeviceptr empty(int size) {
        checkPos(size);
        return Array.empty(size, PrimitiveType.DOUBLE);
    }

    /**
     * Copies contents from a CPU array to a GPU array.
     *
     * @param to The destination GPU array.
     * @param fromArray The source CPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param length The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public static void copy(DArray to, double[] fromArray, int toIndex, int fromIndex, int length) {
        checkNull(fromArray, to);
        Array.copy(to, Pointer.to(fromArray), toIndex, fromIndex, length, PrimitiveType.DOUBLE);
    }

    /**
     * Copies the contents of this GPU array to a CPU array.
     *
     * @param to The destination CPU array.
     * @param toStart The index in the destination array to start copying to.
     * @param fromStart The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or length
     * is negative.
     */
    public void get(double[] to, int toStart, int fromStart, int length) {
        checkNull(to);
        get(Pointer.to(to), toStart, fromStart, length);
    }

    /**
     * Exports a portion of this GPU array to a CPU array.
     *
     * @param fromStart The starting index in this GPU array.
     * @param length The number of elements to export.
     * @return A CPU array containing the exported portion.
     * @throws IllegalArgumentException if fromStart or length is out of bounds.
     */
    public double[] get(int fromStart, int length) {
        double[] export = new double[length];
        get(export, 0, fromStart, length);
        return export;
    }

    /**
     * Exports the entire GPU array to a CPU array.
     *
     * @return A CPU array containing all elements of this GPU array.
     */
    public double[] get() {
        return get(0, length);
    }

    /**
     * Copies a CPU array to this GPU array.
     *
     * @param from The source CPU array.
     * @param toIndex The index in this GPU array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param size The number of elements to copy.
     * @throws IllegalArgumentException if any index is out of bounds or size is
     * negative.
     */
    public void set(double[] from, int toIndex, int fromIndex, int size) {
        copy(this, from, toIndex, fromIndex, size);
    }

    /**
     * Copies a CPU array to this GPU array.
     *
     * @param from The source CPU array.
     * @throws IllegalArgumentException if from is null.
     */
    public final void set(double[] from) {
        set(from, 0, 0, from.length);
    }

    /**
     * Copies a CPU array to this GPU array starting from a specified index.
     *
     * @param from The source CPU array.
     * @param toIndex The index in this GPU array to start copying to.
     * @throws IllegalArgumentException if from is null.
     */
    public void set(double[] from, int toIndex) {
        set(from, toIndex, 0, from.length);
    }


    /**
     * A sub array of this array. Note, this is not a copy and changes to this
     * array will affect the sub array and vice versa.
     *
     * @param start The beginning of the sub array.
     * @param length The length of the sub array.
     * @return
     */
    public DArray subArray(int start, int length) {
        checkPos(start, length);
        return new DArray(pointer(start), length);
    }


    /**
     * Sets the value at the given index.
     *
     * @param index The index the new value is to be assigned to.
     * @param val The new value at the given index.
     */
    public void set(int index, double val) {
        checkPos(index);
        checkAgainstLength(index);
        set(new double[]{val}, index);
    }

    /**
     * Gets the value from the given index.
     *
     * @param index The index the value is to be retrieved from.
     * @return The value at index.
     */
    public DSingleton get(int index) {
        checkPos(index); checkAgainstLength(index);
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
     * @param ldc ldc: Leading dimension of this matrix (ldc ≥ max(1, n)
     * after transposition).
     *
     * @return Status code from CUBLAS library: CUBLAS_STATUS_SUCCESS if the
     * operation was successful, or an appropriate error code otherwise.
     *
     */
    public int MatrixAddWithTranspose(
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
        a.checkAgainstLength(heightA * widthA);
        b.checkAgainstLength(heightA * widthA);
        checkAgainstLength(heightA * widthA);
        
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
     * @param handle
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
    public void outerProd(Handle handle, int rows, int cols, double multProd, DArray vecX, int incX, DArray vecY, int incY) {
        checkNull(handle, vecX, vecY);
        checkPos(rows, cols);
        checkLowerBound(1, incY, incX);
        checkAgainstLength(rows*cols);
        
        JCublas2.cublasDger(handle.get(), rows, cols, cpuPointer(multProd), vecX.pointer, incX, vecY.pointer, incY, pointer, rows);
    }

    /**
     * Computes the Euclidean norm of the vector X (2-norm):
     *
     * <pre>
     * result = sqrt(X[0]^2 + X[1]^2 + ... + X[n-1]^2)
     * </pre>
     *
     * @param handle
     * @return The Euclidean norm of this vector.
     */
    public DSingleton norm(Handle handle) {
        checkNull(handle);
        DSingleton result = new DSingleton();
        JCublas2.cublasDnrm2(handle.get(), length, pointer, 1, result.pointer);
        return result;
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
     * @param handle
     * @param transA Specifies whether matrix A is transposed ('N' for no
     * transpose, 'T' for transpose).
     * @param aRows The number of rows in matrix A.
     * @param aCols The number of columns in matrix A.
     * @param timesAx Scalar multiplier applied to the matrix-vector product.
     * @param matA Pointer to matrix A in GPU memory.
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
    public DArray multMatVec(Handle handle, boolean transA, int aRows, int aCols, double timesAx, DArray matA, DArray vecX, int incX, double beta, int inc) {
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
                aRows,
                vecX.pointer,
                incX,
                cpuPointer(beta),
                pointer,
                inc
        );
        return this;
    }

    @Override
    public String toString() {
        return Arrays.toString(get());
    }

    /**
     * Fills a matrix with a scalar value directly on the GPU using a CUDA
     * kernel.
     *
     * This function sets all elements of the matrix A to the given scalar
     * value. The matrix A is stored in column-major order, and the leading
     * dimension of A is specified by lda.
     *
     * @param fill the scalar value to set all elements of A
     * @param inc The increment with which the method iterates over the array.
     * @return this;
     */
    public DArray fillArray(double fill, int inc) {
        checkLowerBound(1, inc);
        super.fillArray(Pointer.to(new double[]{fill}), inc);
        return this;
    }

    /**
     * Fills a matrix with a value.
     *
     * @param height The height of the matrix.
     * @param width The width of the matrix.
     * @param lda The distance between the first element of each column of the
     * matrix. This should be at least the height of the matrix.
     * @param fill The value the matrix is to be filled with.
     * @return this, after having been filled.
     */
    public DArray fillMatrix(int height, int width, int lda, double fill) {
        checkPos(height, width);
        checkLowerBound(height, lda);
        checkAgainstLength(height * width);
        
        fillMatrix(height, width, lda, Pointer.to(new double[]{fill}));
        return this;
    }

    public static void main(String[] args) {
        DArray test = DArray.emptyArray(6);
//        test.fillArray(0, 1);
        test.fillMatrix(2, 2, 3, 4);

        System.out.println(test);
    }

    
    
    /**
     * Computes the dot product of two vectors:
     *
     * <pre>
     * result = X[0] * Y[0] + X[1] * Y[1] + ... + X[n-1] * Y[n-1]
     * </pre>
     *
     * @param handle
     * @param incX The number of spaces to jump when incrementing forward
     * through x.
     * @param inc The number of spaces to jump when incrementing forward through
     * this array.
     * @param x Pointer to vector X in GPU memory.
     * @return The dot product of X and Y.
     */
    public double dot(Handle handle, DArray x, int incX, int inc) {
        checkNull(handle, x);
        checkLowerBound(1, inc, incX);
        checkUpperBound(length/inc, x.length/incX - 1);
        
        double[] result = new double[1];
        JCublas2.cublasDdot(handle.get(), length, x.pointer, incX, pointer, inc, Pointer.to(result));
        return result[0];
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
     * @param handle
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
        checkUpperBound(length/inc, x.length/incX - 1);

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
        checkLowerBound(height, lda, ldb, ldc);
        checkAgainstLength(height * width);
        
        JCublas2.cublasDgeam(handle.get(),
                transA ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                transB ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N,
                height, width,
                cpuPointer(alpha), a.pointer, lda,
                cpuPointer(beta), b.pointer, ldb,
                pointer, ldc);

        return this;
    }

    /**
     * Scales this vector by the scalar mult:
     *
     * <pre>
     * this = mult * this
     * </pre>
     *
     * @param handle
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
        JCublas2.cublasDscal(handle.get(), length, Pointer.to(new double[]{mult}), pointer, inc);
        return this;
    }
}

//    //TODO: use cuSolver for sovling equations and eigan values and vectors
