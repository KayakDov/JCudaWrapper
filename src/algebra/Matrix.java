package algebra;

import storage.DArray;
import main.GPU;
import processSupport.Handle;
import java.awt.Dimension;
import java.util.Arrays;
import org.apache.commons.math3.exception.*;
import org.apache.commons.math3.linear.*;
import storage.DSingleton;

/**
 * Represents a matrix stored on the GPU. For more information on jcuda
 * http://www.jcuda.org/jcuda/jcublas/JCublas.html
 */
public class Matrix extends AbstractRealMatrix implements AutoCloseable{

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
    private final int colDist; //TODO: make private final

    /**
     * The underlying GPU data storage for this matrix.
     */
    private final DArray data;

    /**
     * Handle for managing JCublas operations, usually unique per thread.
     */
    private Handle handle;

    /**
     * Constructs a new Matrix from a 2D array, where each inner array
     * represents a column of the matrix.
     *
     * @param handle The handle for JCublas operations, required for matrix
     * operations on the GPU.
     * @param matrix A 2D array, where each sub-array is a column of the matrix.
     */
    public Matrix(Handle handle, double[][] matrix) {
        this(matrix[0].length, matrix.length, handle);
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
    public Matrix(DArray array, int height, int width, Handle handle) {
        this(array, height, width, height, handle);
    }

    /**
     * Constructs a new Matrix from an existing RealMatrix object, copying its
     * data to GPU memory.
     *
     * @param mat The matrix to be copied to GPU memory.
     * @param handle The JCublas handle for GPU operations.
     */
    public Matrix(RealMatrix mat, Handle handle) {
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
        this(mat.data, mat.height, mat.width, mat.colDist, mat.handle);
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
    public Matrix(DArray vector, int height, int width, int distBetweenFirstElementOfColumns, Handle handle) {
        if (!GPU.IsAvailable())
            throw new RuntimeException("GPU is not available.");

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
    public Matrix(int height, int width, Handle handle) {
        this(DArray.empty(height * width), height, width, handle);
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
     * Performs matrix multiplication using JCublas.
     *
     * @param m The matrix to multiply with. This matrix is imported to the gpu.
     * @return A new matrix that is the product of this matrix and the other.
     */
    @Override
    public Matrix multiply(RealMatrix m) throws DimensionMismatchException {
        
        Matrix mat = new Matrix(m, handle);
        Matrix result = multiply(mat);
        mat.close(true, false);
        return result;
    }

    /**
     * Performs matrix multiplication using JCublas.
     *
     * @param other The matrix to multiply with.
     * @return A new matrix that is the product of this matrix and the other.
     */
    public Matrix multiply(Matrix other) {
        if (getWidth() != other.getHeight())
            throw new DimensionMismatchException(other.height, width);

        return new Matrix(getHeight(), other.getWidth(), handle)
                .multiplyAndSet(false, false, 1, this, other, 0);
    }

    /**
     * Multiplies two matrices, adding the result into this matrix. The result
     * is inserted into this matrix as a submatrix.
     *
     * @param transposeA True if the first matrix should be transposed.
     * @param transposeB True if the second matrix should be transposed.
     * @param timesAB Scalar multiplier for the product of the two matrices.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param timesThis Scalar multiplier for the elements in this matrix.
     * @return This matrix after the operation.
     */
    public Matrix multiplyAndSet(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {

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
     * Returns the column-major vector index of the given row and column.
     *
     * @param row The row index.
     * @param col The column index.
     * @return The vector index: {@code col * colDist + row}.
     */
    private int index(int row, int col) {
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
        for (int col = 0; col < Math.min(width, matrix.length); col++)
            data.set(handle, matrix[col], index(toRow, toCol + col));
    }

    /**
     * Returns a new matrix that is the sum of this matrix and another matrix.
     *
     * @param m The matrix to be added to this one.
     * @return A new matrix that is the sum of this matrix and another matrix.
     * @throws MatrixDimensionMismatchException If the other matrice's
     * dimensions don't match this matrice's dimensions.
     */
    @Override
    public Matrix add(RealMatrix m) throws MatrixDimensionMismatchException {
        Matrix mat = new Matrix(m, handle);
        Matrix add = add(mat);
        mat.close(true, false);
        return add;
    }

    /**
     * Performs element-wise addition with another matrix.
     *
     * @param other The other matrix to add.
     * @return The result of element-wise addition.
     */
    public Matrix add(Matrix other) {
        if (other.height != height || other.width != width)
            throw new MatrixDimensionMismatchException(other.height, other.width, height, width);

        return new Matrix(height, width, handle).addAndSet(false, false, 1, other, 1, this);
    }

    /**
     * Performs matrix addition or subtraction.
     *
     * <p>
     * This function computes this = alpha * A + beta * B, where A and B are
     * matrices.
     * </p>
     *
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
    public Matrix addAndSet(boolean transA, boolean transB, double alpha, Matrix a, double beta, Matrix b) {

        if (transA) checkRowCol(a.width - 1, a.height - 1);
        else checkRowCol(a.height - 1, a.width - 1);
        if (transB) checkRowCol(b.width - 1, b.height - 1);
        else checkRowCol(b.height - 1, b.width - 1);

        data.addAndSet(handle, transA, transB, height, width, alpha, a.data, a.colDist, beta, b.data, b.colDist, colDist);

        return this;
    }

    /**
     * Subtracts @code{m} from this matrix and returns the difference. This
     * matrix is unchanged.
     *
     * @param m The matrix to me subtracted from this matrix.
     * @return A new matrix that is the difference between the two matrices.
     * @throws MatrixDimensionMismatchException If the dimensions of the
     * matrices don't match.
     */
    @Override
    public Matrix subtract(RealMatrix m) throws MatrixDimensionMismatchException {
        if (m.getRowDimension() != getRowDimension() || m.getColumnDimension() != getColumnDimension())
            throw new MatrixDimensionMismatchException(m.getRowDimension(), m.getColumnDimension(), getRowDimension(), getColumnDimension());
        
        Matrix mat = new Matrix(m, handle);
        Matrix result = subtract(mat);
        mat.close(true, false);
        return result;
    }

    /**
     * Performs element-wise addition with another matrix.
     *
     * @param other The other matrix to add.
     * @return The result of element-wise addition.
     */
    public Matrix subtract(Matrix other) {
        if (getHeight() != other.getHeight() || getWidth() != other.getWidth()) {
            throw new IllegalArgumentException("Matrix dimensions are not compatible for addition");
        }

        return new Matrix(height, width, handle).addAndSet(false, false, -1, other, 1, this);
    }

    /**
     * Multiplies everything in this matrix by a scalar and returns a new
     * matrix. This one remains unchanged.
     *
     * @param d The scalar that does the multiplying.
     * @return A new matrix equal to this matrix times a scalar.
     */
    public Matrix multiply(double d) {
        return new Matrix(height, width, handle).addAndSet(false, false, d, this, 0, this);
    }

    /**
     * Multiplies everything in this matrix by a scalar and returns a new
     * matrix. This one remains unchanged.
     *
     * @param d The scalar that does the multiplying.
     * @return A new matrix equal to this matrix times a scalar.
     */
    @Override
    public Matrix scalarMultiply(double d) {
        return multiply(d);
    }

    /**
     * Fills this matrix with @code{d}, overwriting whatever is there.
     *
     * @param scalar The value to fill the matrix with.
     * @return
     */
    public Matrix fill(double scalar) {
        data.fillMatrix(handle, height, width, colDist, scalar);
        return this;
    }

    /**
     * Adds a scalar value to each element of this matrix and returns a new
     * matrix. The operation does not modify the original matrix.
     *
     * @param d the scalar value to add to each element.
     * @return a new matrix with the scalar value added to each element.
     */
    @Override
    public Matrix scalarAdd(double d) {
        Matrix scalarMat = new Matrix(height, width, handle).fill(d);
        return scalarMat.addAndSet(false, false, 1, this, 1, scalarMat);

    }

    /**
     * Inserts anther matrix into this matrix at the given index.
     *
     * @param other The matrix to be inserted
     * @param row the row in this matrix where the first row of the other matrix
     * is inserted.
     * @param col The column in this matrix where the first row of the other
     * matrix is inserted.
     * @return this.
     *
     */
    public Matrix insert(Matrix other, int row, int col) {
        checkSubMatrixParameters(row, row + other.height, col, col + other.width);

        getSubMatrix(row, row + other.height, col, col + other.width)
                .addAndSet(false, false, 1, other, 0, other);

        return this;
    }

    /**
     * Returns a string representation of the matrix.
     *
     * @return The string representation of the matrix.
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
     * Gets the element at the specified row and column. Note, this is pretty
     * slow.
     *
     * @param row The row index.
     * @param column The column index.
     * @return The element at the specified row and column.
     */
    @Override
    public double getEntry(int row, int column) {
        return data.get(index(row, column)).getVal();

    }

    /**
     * The dimensions of a submatrix.
     *
     * @param startRow The top row of the submatrix.
     * @param endRow The bottom row of the submatrix, inclusive.
     * @param startColumn The first column of a submatrix.
     * @param endColumn The last column of the submatrix, inclusive.
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
        if (startColumn > endColumn)
            throw new NumberIsTooSmallException(endColumn, startColumn, true);
        if (startRow > endRow)
            throw new NumberIsTooSmallException(endRow, startRow, true);

    }

    /**
     * Copies a submatrix defined by the specified rows and columns into a
     * destination 2D array. The submatrix is taken from this matrix and placed
     * in the destination array.
     *
     * @param startRow the starting row index of the submatrix (inclusive).
     * @param endRow the ending row index of the submatrix (exclusive).
     * @param startColumn the starting column index of the submatrix
     * (inclusive).
     * @param endColumn the ending column index of the submatrix (exclusive).
     * @param destination the array into which the submatrix is copied.
     * @throws OutOfRangeException if the specified indices are out of bounds.
     * @throws NumberIsTooSmallException if the submatrix size is too small.
     * @throws MatrixDimensionMismatchException if the destination array
     * dimensions do not match the submatrix dimensions.
     */
    @Override
    public void copySubMatrix(int startRow, int endRow, int startColumn, int endColumn, double[][] destination) throws OutOfRangeException, NumberIsTooSmallException, MatrixDimensionMismatchException {

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        if (destination.length > dim.width)
            throw new MatrixDimensionMismatchException(destination.length, destination[0].length, height, width);

        Matrix subMatrix = getSubMatrix(startRow, endRow, startColumn, endColumn);

        for (int j = 0; j < dim.width; j++)
            destination[j] = subMatrix.getColumn(j);
    }

    /**
     * Returns a submatrix from the specified row and column range. This method
     * passes by reference, meaning changes to the new matrix will affect this
     * matrix and vice versa.
     *
     * @param startRow the starting row index (inclusive).
     * @param endRow the ending row index (exclusive).
     * @param startColumn the starting column index (inclusive).
     * @param endColumn the ending column index (exclusive).
     * @return a new matrix representing the submatrix.
     * @throws OutOfRangeException if the indices are out of bounds.
     * @throws NumberIsTooSmallException if the submatrix dimensions are
     * invalid.
     */
    @Override
    public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        return new Matrix(
                data.subArray(index(startRow, startColumn), dim.width * colDist),
                dim.height,
                dim.width,
                colDist,
                handle
        );
    }

    /**
     * If the row is outside of this matrix, an exception is thrown.
     *
     * @param row The row to be checked.
     * @throws OutOfRangeException
     */
    private void checkRow(int row) throws OutOfRangeException {
        if (row < 0 || row >= height)
            throw new OutOfRangeException(row, 0, height);
    }

    /**
     * If the column is outside of this matrix, an exception is thrown.
     *
     * @param col The column to be checked.
     * @throws OutOfRangeException
     */
    private void checkCol(int col) {
        if (col < 0 || col >= width)
            throw new OutOfRangeException(col, 0, width);
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
        if (Arrays.stream(o).anyMatch(obj -> obj == null))
            throw new NullArgumentException();
    }

    /**
     * Returns a submatrix defined by the specified row and column indices.
     *
     * @param selectedRows the indices of the rows to include.
     * @param selectedColumns the indices of the columns to include.
     * @return a new matrix representing the submatrix.
     * @throws NullArgumentException if any argument is null.
     * @throws NoDataException if the selected rows or columns are empty.
     * @throws OutOfRangeException if any row or column index is out of bounds.
     */
    @Override
    public Matrix getSubMatrix(int[] selectedRows, int[] selectedColumns) throws NullArgumentException, NoDataException, OutOfRangeException {

        checkForNull(selectedRows, selectedColumns);
        if (selectedColumns.length == 0 || selectedRows.length == 0)
            throw new NoDataException();

        Matrix subMat = new Matrix(selectedRows.length, selectedColumns.length, handle);

        int toInd = 0;

        for (int fromColInd : selectedColumns)
            for (int fromRowInd : selectedRows) {
                checkRowCol(fromRowInd, fromColInd);

                subMat.data.set(handle, data, toInd, index(fromRowInd, fromColInd), 1);
            }

        return subMat;
    }

    /**
     * Sets the submatrix at the specified row and column position to the given
     * 2D array values.
     *
     * @param subMatrix the 2D array representing the submatrix.
     * @param row the starting row index.
     * @param column the starting column index.
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
        if (height != other.height || width != other.width) return false;

        return subtract(other).getFrobeniusNorm() <= epsilon;
    }

    /**
     * Creates a new matrix with the specified dimensions.
     *
     * @param height the number of rows.
     * @param width the number of columns.
     * @return a new matrix with the specified dimensions.
     * @throws NotStrictlyPositiveException if the row or column dimension is
     * not positive.
     */
    @Override
    public Matrix createMatrix(int height, int width) throws NotStrictlyPositiveException {
        if (height <= 0 || width <= 0)
            throw new NotStrictlyPositiveException(java.lang.Math.min(height, width));

        return new Matrix(height, width, handle);
    }

    /**
     * Creates a deep copy of this matrix.
     *
     * @return a new matrix that is a copy of this matrix.
     */
    @Override
    public Matrix copy() {
        if (height == colDist)
            return new Matrix(data.copy(handle), height, width, handle);

        Matrix copy = new Matrix(height, width, handle);

        copy.addAndSet(false, false, 1, this, 0, this);

        return copy;
    }

    /**
     * Returns the number of rows in this matrix.
     *
     * @return the number of rows.
     */
    @Override
    public int getRowDimension() {
        return height;
    }

    /**
     * Returns the number of columns in this matrix.
     *
     * @return the number of columns.
     */
    @Override
    public int getColumnDimension() {
        return width;
    }

    /**
     * Sets the entry at the specified row and column to the given value.
     *
     * @param row the row index.
     * @param column the column index.
     * @param value the value to set at the specified position.
     */
    @Override
    public void setEntry(int row, int column, double value) {
        data.set(handle, index(row, column), value);
    }

    /**
     * Returns the row as a new matrix. This method passes by reference, meaning
     * changes to the row matrix will affect this matrix.
     *
     * @param row the index of the row to retrieve.
     * @return a new matrix representing the row.
     * @throws OutOfRangeException if the row index is out of bounds.
     */
    @Override
    public Matrix getRowMatrix(int row) throws OutOfRangeException {
        if (row < 0 || row >= height)
            throw new OutOfRangeException(row, 0, height);

        return new Matrix(
                data.subArray(index(row, 0)),
                1,
                width,
                colDist,
                handle
        );

    }

    /**
     * Returns the row at the specified index as an array of doubles.
     *
     * @param row the row index.
     * @return an array of doubles representing the row.
     * @throws OutOfRangeException if the row index is out of bounds.
     */
    @Override
    public double[] getRow(int row) throws OutOfRangeException {
        return getRow(row, handle);
    }

    /**
     * Returns the row at the specified index as an array of doubles, using the
     * specified handle.
     *
     * @param row the row index.
     * @param handle the handle used for retrieving the row.
     * @return an array of doubles representing the row.
     * @throws OutOfRangeException if the row index is out of bounds.
     */
    public double[] getRow(int row, Handle handle) throws OutOfRangeException {
        if (row < 0 || row >= height)
            throw new OutOfRangeException(row, 0, height);

        Matrix rowMatrix = getRowMatrix(row);
        Matrix rowCopy = rowMatrix.copy();
        double[] rowData = rowCopy.data.get(handle);
        rowCopy.close(true, false);
        return rowData;

    }

    /**
     * Returns the row at the specified index as a RealVector.
     *
     * @param row the row index.
     * @return a RealVector representing the row.
     * @throws OutOfRangeException if the row index is out of bounds.
     */
    @Override
    public RealVector getRowVector(int row) throws OutOfRangeException {//TODO: create a GPU vector class.

        return new ArrayRealVector(getRow(row));
    }

    /**
     * Returns the column at the specified index as an array of doubles.
     *
     * @param column the column index.
     * @return an array of doubles representing the column.
     * @throws OutOfRangeException if the column index is out of bounds.
     */
    @Override
    public double[] getColumn(int column) throws OutOfRangeException {
        if (column >= width || column < 0)
            throw new OutOfRangeException(column, 0, width);

        return getColumnMatrix(column).data.get(handle);
    }

    /**
     * Returns the column as a new matrix. This method passes by reference,
     * meaning changes to the column matrix will affect this matrix and vice
     * versa.
     *
     * @param column the index of the column to retrieve.
     * @return a new matrix representing the column.
     * @throws OutOfRangeException if the column index is out of bounds.
     */
    @Override
    public Matrix getColumnMatrix(int column) throws OutOfRangeException {
        return new Matrix(
                data.subArray(index(0, column), height),
                height,
                1,
                handle
        );

    }

    /**
     * Returns the column at the specified index as a RealVector.
     *
     * @param column the column index.
     * @return a RealVector representing the column.
     * @throws OutOfRangeException if the column index is out of bounds.
     */
    @Override
    public RealVector getColumnVector(int column) throws OutOfRangeException {
        return new ArrayRealVector(getColumn(column));
    }

    /**
     * Returns the data of this matrix as a 2D array. Each column of the matrix
     * is copied into the corresponding element of the array.
     *
     * @return a 2D array representing the matrix data.
     */
    @Override
    public double[][] getData() {
        double[][] getData = new double[width][];
        Arrays.setAll(getData, i -> getColumn(i));
        return getData;
    }

    /**
     * Returns the trace (sum of the diagonal elements) of the matrix. This
     * operation only applies to square matrices.
     *
     * @return the trace of the matrix.
     * @throws NonSquareMatrixException if the matrix is not square.
     */
    @Override
    public double getTrace() throws NonSquareMatrixException {
        if (!isSquare()) throw new NonSquareMatrixException(width, height);

        return data.dot(handle, new DSingleton(1, handle), 0, width + 1);
    }

    /**
     * Returns the hash code of this matrix. The hash code is computed using the
     * data of the matrix.
     *
     * @return the hash code of the matrix.
     */
    @Override
    public int hashCode() {
        return new Array2DRowRealMatrix(getData()).hashCode();
    }

    /**
     * Created a matrix from a doulbe[] representing a column vector.
     *
     * @param vec The column vector.
     * @param handle
     * @return A matrix representing a column vector.
     */
    public static Matrix fromColVec(double[] vec, Handle handle) {
        Matrix mat = new Matrix(vec.length, 1, handle);
        mat.data.set(handle, vec);
        return mat;
    }

    /**
     * Multiplies this matrix by a vector and returns the resulting vector.
     *
     * @param v the vector to be multiplied.
     * @return the resulting vector as an array of doubles.
     * @throws DimensionMismatchException if the vector length does not match
     * the number of columns.
     */
    @Override
    public double[] operate(double[] v) throws DimensionMismatchException {
        if (width != v.length)
            throw new DimensionMismatchException(v.length, width);

        Matrix vec = fromColVec(v, handle);
        Matrix result = multiply(vec);
        vec.close(true, false);

        double[] operate = result.data.get(handle);
        result.close(true, false);

        return operate;
    }

    /**
     * Multiplies this matrix by a RealVector and returns the resulting
     * RealVector.
     *
     * @param v the vector to be multiplied.
     * @return the resulting RealVector.
     * @throws DimensionMismatchException if the vector length does not match
     * the number of columns.
     */
    @Override
    public RealVector operate(RealVector v) throws DimensionMismatchException {
        return new ArrayRealVector(operate(v.toArray()));
    }

    /**
     * The identity Matrix.
     *
     * @param n the height and width of the matrix.
     * @return The identity matrix.
     */
    public static Matrix identity(int n, Handle hand) {

        Matrix ident = new Matrix(n, n, hand);
        ident.data.fill0(hand);
        try (DSingleton one = new DSingleton(1, hand)) {
            ident.data.addToMe(hand, 1, one, 0, n + 1);
        }
        return ident;
    }

    /**
     * Returns the matrix raised to the power of p. This operation only applies
     * to square matrices.
     *
     * @param p the exponent to which the matrix is raised.
     * @return the resulting matrix.
     * @throws NotPositiveException if the exponent is negative.
     * @throws NonSquareMatrixException if the matrix is not square.
     */
    @Override
    public Matrix power(int p) throws NotPositiveException, NonSquareMatrixException {
        if (p < 0) throw new NotPositiveException(p);
        if (!isSquare()) throw new NonSquareMatrixException(width, height);

        if (p == 0) return identity(width, handle);

        if (p % 2 == 0) {
            Matrix halfPow = power(p / 2);
            return halfPow.multiply(halfPow);
        } else return multiply(power(p - 1));
    }

    /**
     * Sets the specified column of the matrix to the values in the given array.
     *
     * @param column the column index.
     * @param array the array of values to set.
     * @throws OutOfRangeException if the column index is out of bounds.
     * @throws MatrixDimensionMismatchException if the length of the array does
     * not match the number of rows.
     */
    @Override
    public void setColumn(int column, double[] array) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkCol(column);
        if (array.length != height)
            throw new MatrixDimensionMismatchException(0, array.length, 0, height);

        data.set(handle, array, index(0, column));
    }

    /**
     * Sets the specified column of the matrix to the values in the given
     * RealMatrix.
     *
     * @param column the column index.
     * @param matrix the RealMatrix to set as the column.
     * @throws OutOfRangeException if the column index is out of bounds.
     * @throws MatrixDimensionMismatchException if the matrix dimensions do not
     * match the column size.
     */
    public void setColumnMatrix(int column, Matrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkCol(column);
        if (matrix.height != height || matrix.width != 1)
            throw new MatrixDimensionMismatchException(matrix.width, matrix.height, 1, height);

        data.set(handle, matrix.data, index(0, column), 0, height);
    }

    /**
     * Sets the specified column of the matrix to the values in the given
     * RealMatrix.
     *
     * @param column the column index.
     * @param matrix the RealMatrix to set as the column.
     * @throws OutOfRangeException if the column index is out of bounds.
     * @throws MatrixDimensionMismatchException if the matrix dimensions do not
     * match the column size.
     */
    @Override
    public void setColumnMatrix(int column, RealMatrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        setColumn(column, matrix.getColumn(0));
    }

    /**
     * Sets the specified column of the matrix to the values in the given
     * RealVector.
     *
     * @param column the column index.
     * @param vector the RealVector to set as the column.
     * @throws OutOfRangeException if the column index is out of bounds.
     * @throws MatrixDimensionMismatchException if the vector length does not
     * match the number of rows.
     */
    @Override
    public void setColumnVector(int column, RealVector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        setColumn(column, vector.toArray());
    }

    /**
     * Sets the specified row of the matrix to the values in the given array.
     *
     * @param row the row index.
     * @param array the array of values to set.
     * @throws OutOfRangeException if the row index is out of bounds.
     * @throws MatrixDimensionMismatchException if the length of the array does
     * not match the number of columns.
     */
    @Override
    public void setRow(int row, double[] array) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkRow(row);
        if (array.length != width)
            throw new MatrixDimensionMismatchException(array.length, 0, width, 0);

        for (int i = 0; i < array.length; i++)
            data.set(handle, array, index(row, i), i, 1);
    }

    /**
     * Sets the row.
     *
     * @param row The index of the row to be set.
     * @param rowMatrix The new values for the row.
     * @throws OutOfRangeException If the row index is illegal.
     * @throws MatrixDimensionMismatchException If the rowMatrix is the wrong
     * size.
     */
    public void setRowMatrix(int row, Matrix rowMatrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        insert(rowMatrix, row, 0);
    }

    /**
     * Sets the row of the matrix using data from another RealMatrix object.
     *
     * @param row the index of the row to set
     * @param matrix the RealMatrix object containing the data for the row
     * @throws OutOfRangeException if the row index is out of range
     * @throws MatrixDimensionMismatchException if the row length does not match
     * the matrix width
     */
    @Override
    public void setRowMatrix(int row, RealMatrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        setRow(row, matrix.getRow(0));
    }

    /**
     * Sets the row of the matrix using data from a RealVector object.
     *
     * @param row the index of the row to set
     * @param vector the RealVector object containing the data for the row
     * @throws OutOfRangeException if the row index is out of range
     * @throws MatrixDimensionMismatchException if the row length does not match
     * the matrix width
     */
    @Override
    public void setRowVector(int row, RealVector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        setRow(row, vector.toArray());
    }

    /**
     * Transposes the current matrix.
     *
     * @return a new Matrix object that is the transpose of this matrix
     */
    @Override
    public Matrix transpose() {
        Matrix transpose = new Matrix(width, height, handle);

        transpose.addAndSet(true, false, 1, this, 0, transpose);

        return transpose;
    }

    /**
     * Applies a RealMatrixChangingVisitor to the matrix elements in
     * column-major order. This method does not use the gpu and is slow.
     *
     * @param visitor the visitor that will modify the matrix elements
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInColumnOrder(RealMatrixChangingVisitor visitor) {
        double[] matrix = data.get(handle);

        Arrays.setAll(matrix, i -> visitor.visit(rowIndex(i), columnIndex(i), matrix[i]));

        data.set(handle, matrix);

        return visitor.end();
    }

    /**
     * Applies a RealMatrixChangingVisitor to a submatrix in column-major order.
     * This method does not use the gpu and is slow.
     *
     * @param visitor the visitor that will modify the matrix elements
     * @param startRow the starting row of the submatrix
     * @param endRow the ending row of the submatrix
     * @param startColumn the starting column of the submatrix
     * @param endColumn the ending column of the submatrix
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInColumnOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) {

        Dimension sub = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        double[] matrix = new double[sub.width * sub.height];

        for (int toCol = 0; toCol <= sub.width; toCol++)
            data.get(handle, matrix, sub.height * toCol, index(0, toCol + startColumn), height);

        Arrays.setAll(matrix, i -> visitor.visit(i % sub.height + startRow, i / sub.height + startColumn, matrix[i]));

        for (int col = startColumn; col <= endColumn; col++)
            data.set(handle, matrix, index(startRow, col), col - startColumn, sub.height);

        return visitor.end();
    }

    /**
     * Applies a RealMatrixChangingVisitor to the matrix elements in an
     * optimized order (currently column-major order). This method does not use
     * the gpu and is slow.
     *
     * @param visitor the visitor that will modify the matrix elements
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixChangingVisitor visitor) {

        return walkInColumnOrder(visitor);
    }

    /**
     * Applies a RealMatrixChangingVisitor to a submatrix in an optimized order
     * (currently column-major order). This method does not use the gpu and is
     * slow.
     *
     * @param visitor the visitor that will modify the matrix elements
     * @param startRow the starting row of the submatrix
     * @param endRow the ending row of the submatrix
     * @param startColumn the starting column of the submatrix
     * @param endColumn the ending column of the submatrix
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) {

        return walkInColumnOrder(visitor, startRow, endRow, startColumn, endColumn);
    }

    /**
     * Applies a RealMatrixPreservingVisitor to the matrix elements in
     * column-major order. This method does not use the gpu and is slow.
     *
     * @param visitor the visitor that will read the matrix elements
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor) {
        return walkInColumnOrder(visitor);
    }

    /**
     * Applies a RealMatrixChangingVisitor to the matrix elements in row-major
     * order. This method is very slow. Column major is faster than this, but is
     * still slow.
     *
     * @param visitor the visitor that will modify the matrix elements
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInRowOrder(RealMatrixChangingVisitor visitor) {
        Matrix transp = transpose();
        double result = transp.walkInColumnOrder(visitor);
        insert(transp, 0, 0);
        return result;

    }

    /**
     * Applies a RealMatrixChangingVisitor to a submatrix in row-major order.
     * This method is in a bad order and does not use the gpu. It is very slow.
     *
     * @param visitor the visitor that will modify the submatrix elements
     * @param startRow the starting row of the submatrix
     * @param endRow the ending row of the submatrix
     * @param startColumn the starting column of the submatrix
     * @param endColumn the ending column of the submatrix
     * @return the result of the visitor's end() method
     * @throws OutOfRangeException if the row or column indices are out of range
     * @throws NumberIsTooSmallException if the submatrix dimensions are invalid
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
     * Applies a RealMatrixPreservingVisitor to the matrix elements in
     * column-major order. This method does not use the gpu and is slow.
     *
     * @param visitor the visitor that will read the matrix elements
     * @return the result of the visitor's end() method
     */
    @Override
    public double walkInColumnOrder(RealMatrixPreservingVisitor visitor) {
        double[] matrix = data.get(handle);

        for (int i = 0; i < matrix.length; i++)
            visitor.visit(i / width, i % width, matrix[i]);

        return visitor.end();
    }

    /**
     * Applies a RealMatrixPreservingVisitor to a submatrix in column-major
     * order. This method does not use the gpu and is slow.
     *
     * @param visitor the visitor that will read the submatrix elements
     * @param startRow the starting row of the submatrix
     * @param endRow the ending row of the submatrix
     * @param startColumn the starting column of the submatrix
     * @param endColumn the ending column of the submatrix
     * @return the result of the visitor's end() method
     * @throws OutOfRangeException if the row or column indices are out of range
     * @throws NumberIsTooSmallException if the submatrix dimensions are invalid
     */
    @Override
    public double walkInColumnOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        Dimension sub = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        double[] matrix = new double[sub.width * sub.height];

        for (int toCol = 0; toCol <= sub.width; toCol++)
            data.get(handle, matrix, toCol * sub.height, index(0, toCol + startColumn), sub.height);

        for (int i = 0; i < matrix.length; i++)
            visitor.visit(i % sub.height + startRow, i / sub.height + startColumn, matrix[i]);

        return visitor.end();
    }

    /**
     * Applies a RealMatrixPreservingVisitor to a submatrix in column-major
     * order. This method does not use the gpu and is slow.
     *
     * @param visitor the visitor that will read the submatrix elements
     * @param startRow the starting row of the submatrix
     * @param endRow the ending row of the submatrix
     * @param startColumn the starting column of the submatrix
     * @param endColumn the ending column of the submatrix
     * @return the result of the visitor's end() method
     * @throws OutOfRangeException if the row or column indices are out of range
     * @throws NumberIsTooSmallException if the submatrix dimensions are invalid
     */
    @Override
    public double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        return walkInColumnOrder(visitor, startRow, endRow, startColumn, endColumn);
    }

    /**
     * Throws an UnsupportedOperationException since this matrix is column-major
     * and cannot be walked in row-major order.
     *
     * @param visitor the visitor that will attempt to read the matrix elements
     * @throws UnsupportedOperationException if row-major order is requested
     */
    @Override
    public double walkInRowOrder(RealMatrixPreservingVisitor visitor) {
        return walkInRowOrder(visitor, 0, height, 0, width);
    }

    /**
     * Throws an UnsupportedOperationException since this matrix is column-major
     * and cannot be walked in row-major order.
     *
     * @param visitor the visitor that will attempt to read the submatrix
     * elements
     * @param startRow the starting row of the submatrix
     * @param endRow the ending row of the submatrix
     * @param startColumn the starting column of the submatrix
     * @param endColumn the ending column of the submatrix
     * @throws UnsupportedOperationException if row-major order is requested
     */
    @Override
    public double walkInRowOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);

        Matrix transpose = getSubMatrix(startRow, endRow, startColumn, endColumn).transpose();
        double result = transpose.walkInColumnOrder(visitor);
        return result;
    }

    /**
     * Computes the Frobenius norm of the matrix. The Frobenius norm is
     * calculated as the square root of the sum of the absolute squares of the
     * matrix's elements.
     *
     * @return The Frobenius norm of the matrix.
     *
     * <p>
     * If the matrix is stored in column-major order with a column distance
     * equal to the matrix height, the operation is performed on the current
     * matrix. Otherwise, a copy of the matrix is created and the operation is
     * performed on the copy.
     * </p>
     */
    @Override
    public double getFrobeniusNorm() {
        Matrix copy = colDist == height ? this : copy();
        return Math.sqrt(copy.data.dot(handle, copy.data, 1, 1));
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
     * Frees resources.
     */
    @Override
    public void close() {
        data.close();
    }

}
