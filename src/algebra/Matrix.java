package algebra;

import array.DArray;
import array.DBatchArray;
import array.DPointerArray;
import resourceManagement.Handle;
import java.awt.Dimension;
import java.util.Arrays;
import org.apache.commons.math3.exception.*;
import org.apache.commons.math3.linear.*;
import array.DSingleton;
import array.IArray;

/**
 * Represents a matrix stored on the GPU. For more information on jcuda
 *
 * TODO: implement extensions of this class to include banded, symmetric banded,
 * triangular packed, symmetric matrices.
 *
 * http://www.jcuda.org/jcuda/jcublas/JCublas.html
 */
public class Matrix extends AbstractRealMatrix implements AutoCloseable {

    /**
     * The number of rows in the matrix.
     */
    private final int height;

    /**
     * The number of columns in the matrix.
     */
    private final int width;

    /**
     * The distance between the first element of each column in memory.
     * <p>
     * Typically, this is equal to the matrix height, but if this matrix is a
     * submatrix, `colDist` may differ, indicating that the matrix data is
     * stored with non-contiguous elements in memory.
     * </p>
     */
    public final int colDist;

    /**
     * The underlying GPU data storage for this matrix.
     */
    protected final DArray data;

    /**
     * Handle for managing JCublas operations, usually unique per thread.
     */
    protected Handle handle;

    /**
     * Constructs a new Matrix from a 2D array, where each inner array
     * represents a column of the matrix.
     *
     * @param handle The handle for JCublas operations, required for matrix
     * operations on the GPU.
     * @param matrix A 2D array, where each sub-array is a column of the matrix.
     */
    public Matrix(Handle handle, double[][] matrix) {
        this(handle, matrix[0].length, matrix.length);
        set(0, 0, matrix);
    }

    /**
     * Constructs a Matrix from a single array representing a column-major
     * matrix.
     *
     * @param array The array storing the matrix in column-major order.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(Handle handle, DArray array, int height, int width) {
        this(handle, array, height, width, height);
    }

    /**
     * Constructs a new Matrix from an existing RealMatrix object, copying its
     * data to GPU memory.
     *
     * @param mat The matrix to be copied to GPU memory.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(Handle handle, RealMatrix mat) {
        this(handle, mat.getData());
    }

    /**
     * Creates a shallow copy of an existing Matrix, referencing the same data
     * on the GPU without copying. Changes to this matrix will affect the
     * original and vice versa.
     *
     * @param mat The matrix to create a shallow copy of.
     */
    public Matrix(Matrix mat) {
        this(mat.handle, mat.data, mat.height, mat.width, mat.colDist);
    }

    /**
     * Constructs a new Matrix from an existing data pointer on the GPU.
     *
     * @param vector Pointer to the data on the GPU.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     * @param distBetweenFirstElementOfColumns The distance between the first
     * element of each column in memory, usually equal to height. If this is a
     * submatrix, it may differ.
     * @param handle The handle for GPU operations.
     */
    public Matrix(Handle handle, DArray vector, int height, int width, int distBetweenFirstElementOfColumns) {
//        if (!GPU.IsAvailable())
//            throw new RuntimeException("GPU is not available.");

        this.height = height;
        this.width = width;
        this.data = vector;
        this.handle = handle;
        this.colDist = distBetweenFirstElementOfColumns;
    }

    /**
     * Constructs an empty matrix of specified height and width.
     *
     * @param handle The handle for GPU operations.
     * @param height The number of rows in the matrix.
     * @param width The number of columns in the matrix.
     */
    public Matrix(Handle handle, int height, int width) {
        this(handle, DArray.empty(height * width), height, width);
    }

    /**
     * Returns the height (number of rows) of the matrix.
     *
     * @return The number of rows in the matrix.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the width (number of columns) of the matrix.
     *
     * @return The number of columns in the matrix.
     */
    public int getWidth() {
        return width;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix multiply(RealMatrix m) throws DimensionMismatchException {

        try (Matrix mat = new Matrix(handle, m)) {
            Matrix result = multiply(mat);
            return result;
        }
    }

    /**
     * Performs matrix multiplication using JCublas.
     *
     * @see Matrix#multiply(org.apache.commons.math3.linear.RealMatrix)
     * @param other The matrix to multiply with.
     * @return A new matrix that is the product of this matrix and the other.
     */
    public Matrix multiply(Matrix other) {
        if (getWidth() != other.getHeight()) {
            throw new DimensionMismatchException(other.height, width);
        }

        return new Matrix(handle, getHeight(), other.getWidth())
                .multiplyAndSet(false, false, 1, this, other, 0);
    }

    /**
     * Multiplies two matrices, adding the result into this matrix. The result
     * is inserted into this matrix as a submatrix.
     *
     * @param handle The handle for this operation.
     * @param transposeA True if the first matrix should be transposed.
     * @param transposeB True if the second matrix should be transposed.
     * @param timesAB Scalar multiplier for the product of the two matrices.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param timesThis Scalar multiplier for the elements in this matrix.
     * @return This matrix after the operation.
     */
    public Matrix multiplyAndSet(Handle handle, boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {

        Dimension aDim = new Dimension(transposeA ? a.height : a.width, transposeA ? a.width : a.height);
        Dimension bDim = new Dimension(transposeB ? b.height : b.width, transposeB ? b.width : b.height);
        Dimension result = new Dimension(bDim.width, aDim.height);

        checkRowCol(result.height - 1, result.width - 1);

        data.multMatMat(handle, transposeA, transposeB,
                aDim.height, bDim.width, aDim.width, timesAB,
                a.data, a.colDist, b.data, b.colDist,
                timesThis, colDist);
        return this;
    }

    /**
     * @see Matrix#multiplyAndSet(processSupport.Handle, boolean, boolean,
     * double, algebra.Matrix, algebra.Matrix, double) timesThis is set to 0,
     * transpose values are false, and timesAB is 1.
     * @param a To be multiplied by the first matrix.
     * @param b To be multiplied by the second matrix.
     * @return This matrix.
     */
    public Matrix multiplyAndSet(Matrix a, Matrix b) {
        return multiplyAndSet(handle, false, false, 1, a, b, 0);
    }

    /**
     * @see Matrix#multiplyAndSet(processSupport.Handle, boolean, boolean,
     * double, algebra.Matrix, algebra.Matrix, double) Uses the default handle.
     * @param transposeA True to transpose A, false otherwise.
     * @param transposeB True to transpose B, false otherwise.
     * @param timesAB TO me multiplied by AB.
     * @param a The A matrix.
     * @param b The B matrix.
     * @param timesThis To be multiplied by this.
     * @return this.
     */
    public Matrix multiplyAndSet(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {
        return multiplyAndSet(handle, transposeA, transposeB, timesAB, a, b, timesThis);
    }

    /**
     * Returns the column-major vector index of the given row and column.
     *
     * @param row The row index.
     * @param col The column index.
     * @return The vector index: {@code col * colDist + row}.
     */
    protected int index(int row, int col) {
        return col * colDist + row;
    }

    /**
     * Returns the row index corresponding to the given column-major vector
     * index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The row index.
     */
    private int rowIndex(int vectorIndex) {
        return vectorIndex % height;
    }

    /**
     * Returns the column index corresponding to the given column-major vector
     * index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The column index.
     */
    private int columnIndex(int vectorIndex) {
        return vectorIndex / height;
    }

    /**
     * Copies a matrix to the GPU and stores it in the internal data structure.
     *
     * @param toRow The starting row index in this matrix.
     * @param toCol The starting column index in this matrix.
     * @param matrix The matrix to be copied, represented as an array of
     * columns.
     */
    private final void set(int toRow, int toCol, double[][] matrix) {
        for (int col = 0; col < Math.min(width, matrix.length); col++) {
            data.set(handle, matrix[col], index(toRow, toCol + col));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix add(RealMatrix m) throws MatrixDimensionMismatchException {
        try (Matrix mat = new Matrix(handle, m)) {
            return add(mat);
        }

    }

    /**
     * @see Matrix#add(org.apache.commons.math3.linear.RealMatrix)
     * @param other The other matrix to add.
     * @return A new matrix, the result of element-wise addition.
     */
    public Matrix add(Matrix other) {
        if (other.height != height || other.width != width) {
            throw new MatrixDimensionMismatchException(other.height, other.width, height, width);
        }

        return new Matrix(handle, height, width).addAndSet(1, other, 1, this);
    }

    /**
     * Performs matrix addition or subtraction.
     *
     * <p>
     * This function computes this = alpha * A + beta * B, where A and B are
     * matrices.
     * </p>
     *
     * @param handle The handle.
     * @param transA specifies whether matrix A is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param transB specifies whether matrix B is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @return this
     *
     */
    public Matrix addAndSet(Handle handle, boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {

        if (transA) {
            checkRowCol(a.width - 1, a.height - 1);
        } else {
            checkRowCol(a.height - 1, a.width - 1);
        }
        if (transB) {
            checkRowCol(b.width - 1, b.height - 1);
        } else {
            checkRowCol(b.height - 1, b.width - 1);
        }

        data.addAndSet(handle, transA, transB, height, width, alpha, a.data, a.colDist, beta, b.data, b.colDist, colDist);

        return this;
    }

    /**
     * @see Matrix#addAndSet(boolean, boolean, double, algebra.Matrix, double,
     * algebra.Matrix) Uses default handle.
     * @param transA True to transpose A.
     * @param transB True to transpose B.
     * @param alpha the multiply by A.
     * @param a The A matrix.
     * @param beta To multiply by B.
     * @param b The B matrix.
     * @return This.
     */
    public Matrix addAndSet(boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {
        return addAndSet(handle, transA, transB, alpha, a, beta, b);
    }

    /**
     * @see Matrix#addAndSet(boolean, boolean, double, algebra.Matrix, double,
     * algebra.Matrix) Uses default handle.
     * @param alpha the multiply by A.
     * @param a The A matrix.
     * @param beta To multiply by B.
     * @param b The B matrix.
     * @return This.
     */
    public Matrix addAndSet(double alpha, Matrix a, double beta, Matrix b) {
        return addAndSet(handle, false, false, alpha, a, beta, b);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix subtract(RealMatrix m) throws MatrixDimensionMismatchException {
        if (m.getRowDimension() != getRowDimension() || m.getColumnDimension() != getColumnDimension()) {
            throw new MatrixDimensionMismatchException(m.getRowDimension(), m.getColumnDimension(), getRowDimension(), getColumnDimension());
        }

        try (Matrix mat = new Matrix(handle, m)) {
            return subtract(mat);
        }
    }

    /**
     * @see Matrix#subtract(org.apache.commons.math3.linear.RealMatrix)
     *
     * @param other The other matrix to add.
     * @return The result of element-wise addition.
     */
    public Matrix subtract(Matrix other) {
        if (getHeight() != other.getHeight() || getWidth() != other.getWidth()) {
            throw new IllegalArgumentException("Matrix dimensions are not compatible for addition");
        }

        return new Matrix(handle, height, width).addAndSet(-1, other, 1, this);
    }

    /**
     * Multiplies everything in this matrix by a scalar and returns a new
     * matrix. This one remains unchanged.
     *
     * @param d The scalar that does the multiplying.
     * @return A new matrix equal to this matrix times a scalar.
     */
    public Matrix multiply(double d) {
        return new Matrix(handle, height, width).addAndSet(d, this, 0, this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix scalarMultiply(double d) {
        return multiply(d);
    }

    /**
     * Fills this matrix with @code{d}, overwriting whatever is there.
     *
     * @param scalar The value to fill the matrix with.
     * @return this.
     */
    public Matrix fill(double scalar) {
        data.fillMatrix(handle, height, width, colDist, scalar);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix scalarAdd(double d) {
        Matrix scalarMat = new Matrix(handle, height, width).fill(d);
        return scalarMat.addAndSet(1, this, 1, scalarMat);

    }

    /**
     * Inserts anther matrix into this matrix at the given index.
     *
     * @param handle The handle with which to perform this operation.
     * @param other The matrix to be inserted
     * @param row the row in this matrix where the first row of the other matrix
     * is inserted.
     * @param col The column in this matrix where the first row of the other
     * matrix is inserted.
     * @return this.
     *
     */
    public Matrix insert(Handle handle, Matrix other, int row, int col) {
        checkSubMatrixParameters(row, row + other.height, col, col + other.width);

        getSubMatrix(row, row + other.height, col, col + other.width)
                .addAndSet(1, other, 0, other);

        return this;
    }

    /**
     * @see Matrix#insert(processSupport.Handle, algebra.Matrix, int, int)
     * except with default handle.
     *
     * @param other
     * @param row
     * @param col
     * @return
     */
    public Matrix insert(Matrix other, int row, int col) {
        return insert(handle, other, row, col);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int row = 0; row < getHeight(); row++) {
            sb.append("[");
            for (int col = 0; col < getWidth(); col++) {
                sb.append(getEntry(row, col));
                if (col < getWidth() - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
            if (row < getHeight() - 1) {
                sb.append(",\n ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getEntry(int row, int column) {
        return data.get(index(row, column)).getVal(handle);

    }

    /**
     * The dimensions of a submatrix.
     *
     * @param startRow The top row of the submatrix.
     * @param endRow The bottom row of the submatrix, exclusive.
     * @param startColumn The first column of a submatrix.
     * @param endColumn The last column of the submatrix, exclusive.
     * @return The dimensions of a submatrix.
     */
    private Dimension subMatrixDimensions(int startRow, int endRow, int startColumn, int endColumn) {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);
        return new Dimension(endColumn - startColumn, endRow - startRow);
    }

    /**
     * Does some basic checks on the validity of the subMatrix parameters. Throw
     * exceptions if there are any problems.
     *
     * @param startRow inclusive
     * @param endRow exclusive
     * @param startColumn inclusive
     * @param endColumn exclusive
     * @throws OutOfRangeException
     * @throws NumberIsTooSmallException
     */
    private void checkSubMatrixParameters(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkRowCol(endRow - 1, endColumn - 1);
        checkRowCol(startRow, startColumn);
        if (startColumn > endColumn) {
            throw new NumberIsTooSmallException(endColumn, startColumn, true);
        }
        if (startRow > endRow) {
            throw new NumberIsTooSmallException(endRow, startRow, true);
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void copySubMatrix(int startRow, int endRow, int startColumn, int endColumn, double[][] destination) throws OutOfRangeException, NumberIsTooSmallException, MatrixDimensionMismatchException {

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        if (destination.length > dim.width) {
            throw new MatrixDimensionMismatchException(destination.length, destination[0].length, height, width);
        }

        Matrix subMatrix = getSubMatrix(startRow, endRow, startColumn, endColumn);

        for (int j = 0; j < dim.width; j++) {
            destination[j] = subMatrix.getColumn(j);
        }
    }

    /**
     * Passes by reference. Changes to the sub matrix will effect the original
     * matrix and vice versa.
     *
     * {@inheritDoc}
     */
    @Override
    public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        return new Matrix(
                handle,
                data.subArray(index(startRow, startColumn), dim.width * colDist),
                dim.height,
                dim.width,
                colDist);
    }

    /**
     * If the row is outside of this matrix, an exception is thrown.
     *
     * @param row The row to be checked.
     * @throws OutOfRangeException
     */
    private void checkRow(int row) throws OutOfRangeException {
        if (row < 0 || row >= height) {
            throw new OutOfRangeException(row, 0, height);
        }
    }

    /**
     * If the column is outside of this matrix, an exception is thrown.
     *
     * @param col The column to be checked.
     * @throws OutOfRangeException
     */
    private void checkCol(int col) {
        if (col < 0 || col >= width) {
            throw new OutOfRangeException(col, 0, width);
        }
    }

    /**
     * If either the row or column are out of range, an exception is thrown.
     *
     * @param row The row to be checked.
     * @param col The column to be checked.
     */
    private void checkRowCol(int row, int col) throws OutOfRangeException {
        checkRow(row);
        checkCol(col);
    }

    /**
     * Checks if any of the objects passed are null, and if they are, throws a
     * null argument exception.
     *
     * @param o To be checked for null values.
     */
    private void checkForNull(Object... o) {
        if (Arrays.stream(o).anyMatch(obj -> obj == null)) {
            throw new NullArgumentException();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix getSubMatrix(int[] selectedRows, int[] selectedColumns) throws NullArgumentException, NoDataException, OutOfRangeException {

        checkForNull(selectedRows, selectedColumns);
        if (selectedColumns.length == 0 || selectedRows.length == 0) {
            throw new NoDataException();
        }

        Matrix subMat = new Matrix(handle, selectedRows.length, selectedColumns.length);

        int toInd = 0;

        for (int fromColInd : selectedColumns) {
            for (int fromRowInd : selectedRows) {
                checkRowCol(fromRowInd, fromColInd);

                subMat.data.set(handle, data, toInd, index(fromRowInd, fromColInd), 1);
            }
        }

        return subMat;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setSubMatrix(double[][] subMatrix, int row, int column) {
        set(row, column, subMatrix);
    }

    /**
     * The number of elements in this matrix.
     *
     * @return The number of elements in this matrix.
     */
    public int size() {
        return width * height;
    }

    /**
     * Checks to see if the two matrices are equal to within a margin of 1e-10.
     *
     * @param object
     * @return True if they are equal, false otherwise.
     */
    public boolean equals(Matrix object) {
        return equals(object, 1e-10);
    }

    /**
     * Checks if the two methods are equal to within an epsilon margin of error.
     *
     * @param other A matrix that might be equal to this one.
     * @param epsilon The acceptable margin of error.
     * @return True if the matrices are very close to one another, false
     * otherwise.
     */
    public boolean equals(Matrix other, double epsilon) {
        if (height != other.height || width != other.width) {
            return false;
        }

        return subtract(other).getFrobeniusNorm() <= epsilon;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix createMatrix(int height, int width) throws NotStrictlyPositiveException {
        if (height <= 0 || width <= 0) {
            throw new NotStrictlyPositiveException(java.lang.Math.min(height, width));
        }

        return new Matrix(handle, height, width);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix copy() {
        if (height == colDist) {
            return new Matrix(handle, data.copy(handle), height, width);
        }

        Matrix copy = new Matrix(handle, height, width);

        copy.addAndSet(1, this, 0, this);

        return copy;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getRowDimension() {
        return height;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getColumnDimension() {
        return width;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setEntry(int row, int column, double value) {
        data.set(handle, index(row, column), value);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix getRowMatrix(int row) throws OutOfRangeException {
        if (row < 0 || row >= height) {
            throw new OutOfRangeException(row, 0, height);
        }

        return new Matrix(
                handle, data.subArray(index(row, 0)),
                1,
                width,
                colDist);

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] getRow(int row) throws OutOfRangeException {
        return getRow(row, handle);
    }

    /**
     * @see Matrix#getRow(int)
     */
    public double[] getRow(int row, Handle handle) throws OutOfRangeException {
        if (row < 0 || row >= height) {
            throw new OutOfRangeException(row, 0, height);
        }

        Matrix rowMatrix = getRowMatrix(row);

        try (Matrix rowCopy = rowMatrix.copy()) {
            return rowCopy.data.get(handle);
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector getRowVector(int row) throws OutOfRangeException {//TODO: create a GPU vector class.

        return new Vector(handle, data.subArray(row), colDist);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] getColumn(int column) throws OutOfRangeException {
        if (column >= width || column < 0) {
            throw new OutOfRangeException(column, 0, width);
        }

        return getColumnMatrix(column).data.get(handle);
    }

    /**
     * This method passes by reference, meaning changes to the column matrix
     * will affect this matrix and vice versa. {@inheritDoc}
     */
    @Override
    public Matrix getColumnMatrix(int column) throws OutOfRangeException {
        return new Matrix(
                handle, data.subArray(index(0, column), height),
                height,
                1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector getColumnVector(int column) throws OutOfRangeException {
        return new Vector(handle, data.subArray(index(0, column), height), 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[][] getData() {
        double[][] getData = new double[width][];
        Arrays.setAll(getData, i -> getColumn(i));
        return getData;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getTrace() throws NonSquareMatrixException {
        if (!isSquare()) {
            throw new NonSquareMatrixException(width, height);
        }

        return data.dot(handle, new DSingleton(handle, 1), 0, width + 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int hashCode() {
        return new Array2DRowRealMatrix(getData()).hashCode();
    }

    /**
     * Created a matrix from a double[] representing a column vector.
     *
     * @param vec The column vector.
     * @param handle
     * @return A matrix representing a column vector.
     */
    public static Matrix fromColVec(double[] vec, Handle handle) {
        Matrix mat = new Matrix(handle, vec.length, 1);
        mat.data.set(handle, vec);
        return mat;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] operate(double[] v) throws DimensionMismatchException {
        if (width != v.length) {
            throw new DimensionMismatchException(v.length, width);
        }

        Matrix vec = fromColVec(v, handle);
        Matrix result = multiply(vec);
        vec.close();

        double[] operate = result.data.get(handle);
        result.close();

        return operate;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector operate(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return operate(temp);
        }
    }

    /**
     * @see Matrix#operate(org.apache.commons.math3.linear.RealVector)
     */
    public Vector operate(Vector v) throws DimensionMismatchException {
        Vector result = new Vector(handle, height);
        result.dArray().multMatVec(handle, false, height, width, 1, data, colDist, v.dArray(), v.inc, 1, 1);
        return result;
    }

    /**
     * The identity Matrix.
     *
     * @param n the height and width of the matrix.
     * @param hand
     * @return The identity matrix.
     */
    public static Matrix identity(int n, Handle hand) {

        Matrix ident = new Matrix(hand, n, n);
        ident.data.fill0(hand);
        try (DSingleton one = new DSingleton(hand, 1)) {
            ident.data.addToMe(hand, 1, one, 0, n + 1);
        }
        return ident;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix power(int p) throws NotPositiveException, NonSquareMatrixException {
        if (p < 0) {
            throw new NotPositiveException(p);
        }
        if (!isSquare()) {
            throw new NonSquareMatrixException(width, height);
        }

        if (p == 0) {
            return identity(width, handle);
        }

        if (p % 2 == 0) {
            Matrix halfPow = power(p / 2);
            return halfPow.multiply(halfPow);
        } else {
            return multiply(power(p - 1));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setColumn(int column, double... array) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkCol(column);
        if (array.length != height) {
            throw new MatrixDimensionMismatchException(0, array.length, 0, height);
        }

        data.set(handle, array, index(0, column));
    }

    /**
     * {@inheritDoc}
     */
    public void setColumnMatrix(int column, Matrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkCol(column);
        if (matrix.height != height || matrix.width != 1) {
            throw new MatrixDimensionMismatchException(matrix.width, matrix.height, 1, height);
        }

        data.set(handle, matrix.data, index(0, column), 0, height);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setColumnMatrix(int column, RealMatrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        setColumn(column, matrix.getColumn(0));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setColumnVector(int column, RealVector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        setColumn(column, vector.toArray());
    }

    /**
     * @see Matrix#setColumnVector(int,
     * org.apache.commons.math3.linear.RealVector)
     */
    public void setColumnVector(int column, Vector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        data.set(handle, vector.dArray(), index(0, column), 0, 0, vector.inc, Math.min(height, vector.getDimension()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setRow(int row, double[] array) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkRow(row);
        if (array.length != width) {
            throw new MatrixDimensionMismatchException(array.length, 0, width, 0);
        }

        try (Vector temp = new Vector(handle, array)) {
            setRowVector(row, temp);
        }
    }

    /**
     * @see Matrix#setRowMatrix(int, org.apache.commons.math3.linear.RealMatrix)
     */
    public void setRowMatrix(int row, Matrix rowMatrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        insert(rowMatrix, row, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setRowMatrix(int row, RealMatrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        setRow(row, matrix.getRow(0));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setRowVector(int row, RealVector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        setRow(row, vector.toArray());
    }

    /**
     * @see Matrix#setRowVector(int, org.apache.commons.math3.linear.RealVector)
     */
    public void setRowVector(int row, Vector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        data.set(handle, vector.dArray(), index(row, 0), 0, colDist, vector.inc, Math.min(width, vector.getDimension()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix transpose() {
        Matrix transpose = new Matrix(handle, width, height);

        transpose.addAndSet(true, false, 1, this, 0, transpose);

        return transpose;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInColumnOrder(RealMatrixChangingVisitor visitor) {
        double[] matrix = data.get(handle);

        visitor.start(height, width, 0, height, 0, width);

        Arrays.setAll(matrix, i -> visitor.visit(rowIndex(i), columnIndex(i), matrix[i]));

        data.set(handle, matrix);

        return visitor.end();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInColumnOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) {

        Dimension sub = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        double[] matrix = new double[sub.width * sub.height];

        for (int toCol = 0; toCol <= sub.width; toCol++) {
            data.get(handle, matrix, sub.height * toCol, index(0, toCol + startColumn), height);
        }

        Arrays.setAll(matrix, i -> visitor.visit(i % sub.height + startRow, i / sub.height + startColumn, matrix[i]));

        for (int col = startColumn; col <= endColumn; col++) {
            data.set(handle, matrix, index(startRow, col), col - startColumn, sub.height);
        }

        return visitor.end();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixChangingVisitor visitor) {

        return walkInColumnOrder(visitor);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) {

        return walkInColumnOrder(visitor, startRow, endRow, startColumn, endColumn);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor) {
        return walkInColumnOrder(visitor);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInRowOrder(RealMatrixChangingVisitor visitor) {
        Matrix transp = transpose();
        double result = transp.walkInColumnOrder(visitor);
        insert(transp, 0, 0);
        return result;

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInRowOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);

        Matrix transpose = getSubMatrix(startRow, endRow, startColumn, endColumn).transpose();
        double result = transpose.walkInColumnOrder(visitor);
        insert(transpose, startRow, startColumn);
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInColumnOrder(RealMatrixPreservingVisitor visitor) {
        double[] matrix = data.get(handle);

        visitor.start(height, width, 0, height, 0, width);
        for (int i = 0; i < matrix.length; i++) {
            visitor.visit(i / width, i % width, matrix[i]);
        }

        return visitor.end();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInColumnOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        Dimension sub = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        double[] matrix = new double[sub.width * sub.height];

        visitor.start(height, width, 0, height, 0, width);

        for (int toCol = 0; toCol <= sub.width; toCol++) {
            data.get(handle, matrix, toCol * sub.height, index(0, toCol + startColumn), sub.height);
        }

        for (int i = 0; i < matrix.length; i++) {
            visitor.visit(i % sub.height + startRow, i / sub.height + startColumn, matrix[i]);
        }

        return visitor.end();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        return walkInColumnOrder(visitor, startRow, endRow, startColumn, endColumn);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInRowOrder(RealMatrixPreservingVisitor visitor) {
        return walkInRowOrder(visitor, 0, height, 0, width);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double walkInRowOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);

        Matrix transpose = getSubMatrix(startRow, endRow, startColumn, endColumn).transpose();
        double result = transpose.walkInColumnOrder(visitor);
        return result;
    }

    /**
     * A vector containing the dot product of each column and itself.
     *
     * @return A vector containing the dot product of each column and itself.
     */
    public Vector columnsSquared() {        

        MatrixStride columns = new MatrixStride(handle, data.getStrided(height), height);
        
        return new MatrixStride(handle, 1, width)
                .multAndAdd(true, false, columns, columns, 1, 0)
                .asVector();
    }

    /**
     * <p>
     * If the matrix is stored in column-major order with a column distance
     * equal to the matrix height, the operation is performed on the current
     * matrix. Otherwise, a copy of the matrix is created and the operation is
     * performed on the copy.
     * </p>
     *
     * {@inheritDoc}
     */
    @Override
    public double getFrobeniusNorm() {

        try (Vector colsSq = columnsSquared()) {
            return Math.sqrt(colsSq.getL1Norm());
        }

    }

    /**
     * There should be one handle per thread.
     *
     * @param handle The handle used by this matrix.
     */
    public void setHandle(Handle handle) {
        this.handle = handle;
    }

    /**
     * There should be one handle per thread.
     *
     * @return The handle used by this matrix.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        if (colDist != height) {
            throw new IllegalAccessError("You are cleaning data from a sub Matrix");
        }
        data.close();
    }

    /**
     * Adds to this matrix the outer product of a and b.
     *
     * @param a A column.
     * @param b A row
     * @return The outer product of a and b.
     */
    public Matrix addOuterProduct(Vector a, Vector b) {

        data.outerProd(handle, height, width, 1, a.dArray(), a.inc, b.dArray(), b.inc, colDist);

        return this;
    }

//    
//    /**
//     * A copy of this matrix as a vector in column major order.
//     * @return A copy of this matrix as a vector in column major order.
//     */
//    public Vector asVectorCopy(){
//        return new Vector(handle, copy().data, 1);
//    }
//    
//    
    /**
     * An unmodifiable version of this matrix.
     *
     * @return An unmodifiable version of this matrix.
     */
    public unmodifiable.Matrix unmodifable() {
        return new unmodifiable.Matrix(handle, data, height, width, colDist);
    }

    /**
     * The underlying column major data.
     *
     * @return The underlying column major data.
     */
    public DArray dArray() {
        return data;
    }

    /**
     * The underlying data of this matrix in a vector.
     *
     * @return The underlying data of this matrix in a vector.
     */
    public Vector asVector() {
        return new Vector(handle, data, 1);
    }

    /**
     * Creates a matrix from the underlying data in this matrix with a new
     * height, width, and distance between columns. Note that if the distance
     * between columns in the new matrix is less that in this matrix, the matrix
     * will contain data that this one does not.
     *
     * @param newHieght The height of the new matrix.
     * @param newWidth The width of the new matrix.
     * @param newColDist The distance between columns of the new matrix. By
     * setting the new column distance to be less than or greater than the old
     * one, the new matrix may have more or fewer elements.
     * @return A shallow copy of this matrix that has a different shape.
     *
     */
    public Matrix newDimensions(int newHieght, int newWidth, int newColDist) {
        return new Matrix(handle, data, newHieght, newWidth, newColDist);
    }

    /**
     * Creates a matrix from the underlying data in this matrix with a new
     * height, width, and distance between columns. Note that if the distance
     * between columns in the new matrix is less that in this matrix, the matrix
     * will contain data that this one does not.
     *
     * @param newHieght The height of the new matrix. The width*colDist should
     * be divisible by this number.
     * @return A shallow copy of this matrix that has a different shape.
     *
     */
    public Matrix newDimensions(int newHieght) {
        return newDimensions(newHieght, size() / newHieght, newHieght);
    }

    /**
     * The distance between the 1st element of each column in column major
     * order.
     *
     * @return The distance between the first element of each column in column
     * major order.
     */
    public int getColDist() {
        return colDist;
    }


}
