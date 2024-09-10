package algebra;

import gpu.DArray;
import main.GPU;
import gpu.Handle;
import java.awt.Dimension;
import java.awt.Point;
import java.util.Arrays;
import jcuda.jcublas.*;
import org.apache.commons.math3.exception.*;
import org.apache.commons.math3.linear.*;

/**
 * Represents a matrix stored on the GPU. For more information on jcuda
 * http://www.jcuda.org/jcuda/jcublas/JCublas.html
 */
public class Matrix extends AbstractRealMatrix {

    private final int height;
    private final int width;

    /**
     * The distance between the first element of each column. This will usually
     * be the height, but if this matrix is a sub matrix, it may not be.
     */
    private final int colDist;

    private final DArray data;

    private Handle handle;

    /**
     * This is a column major matrix. Each inner array in the parameter is a
     * column of the matrix created.
     *
     * @param matrix
     */
    public Matrix(double[][] matrix, Handle handle) {
        this(handle, matrix[0].length, matrix.length);
        set(matrix);
    }

    /**
     *
     * @param array A single array describing a column major matrix. Each column
     * is stored in order in the array.
     * @param numRows The number of rows in the matrix.
     * @param numColumns The number of columns in the matrix.
     * @param handle There should be one handle per thread.
     */
    public Matrix(DArray array, int numRows, int numColumns, Handle handle) {
        this(array, numRows, numColumns, numRows, handle);
    }

    /**
     * Constructor that converts a RealMatrix to a GPUMatrix
     *
     * @param mat matrix to be copied to gpu memory.
     * @param handle One per thread.
     */
    public Matrix(RealMatrix mat, Handle handle) {
        this(mat.getData(), handle);
    }

    /**
     * This matrix now has the same pointer as the proffered matrix. There is no
     * deep copy. Changes to this matrix will be felt in the other matrix. Use
     * copy method to get a copy.
     *
     * @param mat
     */
    public Matrix(Matrix mat) {
        this(mat.data, mat.height, mat.width, mat.colDist, mat.handle);
    }

    /**
     * Constructs a matrix from a Pointer to existing data on the GPU.
     *
     * @param vector Pointer to data on the GPU.
     * @param height The height of the matrix.
     * @param width The width of the matrix.
     * @param distBetweenFirstElementOfCOlumns The distance between the first
     * element of each column. This will usually be the height, but if this
     * matrix is a sub matrix, it may not be.
     */
    Matrix(DArray vector, int height, int width, int distBetweenFirstElementOfCOlumns, Handle handle) {
        if (!GPU.IsAvailable())
            throw new RuntimeException("GPU is not available.");

        this.height = height;
        this.width = width;
        this.data = vector;
        this.handle = handle;
        this.colDist = distBetweenFirstElementOfCOlumns;

    }

    /**
     * An empty matrix.
     *
     * @param height The height of the empty matrix.
     * @param width The width of the empty matrix.
     */
    public Matrix(Handle handle, int height, int width) {
        this(DArray.emptyArray(height * width), height, width, handle);
    }

    /**
     * Gets the height of the matrix.
     *
     * @return The height of the matrix.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Gets the width of the matrix.
     *
     * @return The width of the matrix.
     */
    public int getWidth() {
        return width;
    }

    @Override
    public Matrix multiply(RealMatrix m) throws DimensionMismatchException {
        return multiply(new Matrix(m, handle));
    }

    /**
     * Performs matrix multiplication with another matrix using JCublas.
     *
     * @param other The other matrix to multiply with.
     * @return The result of matrix multiplication.
     */
    public Matrix multiply(Matrix other) {
        if (getWidth() != other.getHeight())
            throw new DimensionMismatchException(other.height, width);

        return new Matrix(handle, getHeight(), other.getWidth())
                .multiplyAndSet(false, false, 1, this, other, 0);
    }

    /**
     * Multiplies two matrices and adds the product into this matrix as a sub
     * matrix.
     *
     * @param transposeA True if A is to be transposed before multiplication.
     * False otherwise.
     * @param transposeB True if b is to be transposed before multiplication.
     * false Otherwise.
     * @param timesAB A scalar to multiply the product by.
     * @param a The first matrix to be multiplied.
     * @param b The second matrix to be multiplied.
     * @param timesThis To multiply the elements in this matrix at the insertion
     * site before the product is added to them.
     * @return This.
     */
    public Matrix multiplyAndSet(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {

        Dimension aDim = new Dimension(transposeA ? a.height : a.width, transposeA ? a.width : a.height);
        Dimension bDim = new Dimension(transposeB ? b.height : b.width, transposeB ? b.width : b.height);
        Dimension result = new Dimension(bDim.width, aDim.height);

        checkRowCol(result.height, result.width);

        data.multMatMat(handle, transposeA, transposeB,
                aDim.height, bDim.width, aDim.width, timesAB,
                a.data, a.colDist, b.data, b.colDist,
                timesThis, colDist);

        return this;
    }

    /**
     * The column-major vector index of the requested location.
     *
     * @param row The row index of the desired vector index.
     * @param col The column index of the desired index.
     * @return The vector index: col * height + row;
     */
    private int index(int row, int col) {
        return col * height + row;
    }

    /**
     * The matrix row index of the proffered column-major vector index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The row in this matrix.
     */
    private int rowIndex(int vectorIndex) {
        return vectorIndex % height;
    }

    /**
     * The matrix column index of the proffered column-major vector index.
     *
     * @param vectorIndex The index in the underlying storage vector.
     * @return The column in this matrix.
     */
    private int columnIndex(int vectorIndex) {
        return vectorIndex / height;
    }

    /**
     * Copies a matrix to the gpu and returns a pointer to the matrix.
     *
     * @param matrix Gets copied to the gpu. It is an array of columns.
     * @return A pointer to the matrix.
     */
    private void set(double[][] matrix) {

        for (int col = 0; col < matrix[0].length; col++)
            data.set(matrix[col], index(0, col));
    }

    @Override
    public Matrix add(RealMatrix m) throws MatrixDimensionMismatchException {
        return add(new Matrix(m, handle));
    }

    /**
     * Performs element-wise addition with another matrix.
     *
     * @param other The other matrix to add.
     * @return The result of element-wise addition.
     */
    public Matrix add(Matrix other) {
        if (other.getRowDimension() != getRowDimension() || other.getColumnDimension() != getColumnDimension())
            throw new MatrixDimensionMismatchException(other.getRowDimension(), other.getColumnDimension(), getRowDimension(), getColumnDimension());

        return new Matrix(handle, height, width).addAndSet(false, false, 1, other, 1, this);
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

        checkRowCol(a.height, a.width);
        checkRowCol(b.height, b.width);

        data.addAndSet(handle, transA, transB, height, width, alpha, a.data, a.colDist, beta, b.data, b.colDist, colDist);

        return this;
    }

    @Override
    public Matrix subtract(RealMatrix m) throws MatrixDimensionMismatchException {
        if (m.getRowDimension() != getRowDimension() || m.getColumnDimension() != getColumnDimension())
            throw new MatrixDimensionMismatchException(m.getRowDimension(), m.getColumnDimension(), getRowDimension(), getColumnDimension());
        return subtract(new Matrix(m, handle));
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

        return new Matrix(handle, height, width).addAndSet(false, false, -1, other, 1, this);
    }

    /**
     * Multiplies everything in this matrix by a scalar and returns a new
     * matrix. This one remains unchanged.
     *
     * @param scalar The scalar that does the multiplying.
     * @return A new matrix equal to this matrix times a scalar.
     */
    public Matrix multiply(double scalar) {
        return copy().multiplyAndSet(false, false, 0, null, null, scalar);
    }

    @Override
    public Matrix scalarMultiply(double d) {
        return multiply(d);
    }

    @Override
    public Matrix scalarAdd(double d) {
        return new Matrix(handle, height, width).
    }

    /**
     * Inserts anther matrix into this matrix at the given index.
     *
     * @param other The matrix to be inserted
     * @param row the row in this matrix where the first row of the other matrix
     * is inserted.
     * @param col The column in this matrix where the first row of the other
     * matrix is inserted.
     *
     */
    public void insert(Matrix other, int row, int col) {
        for (int i = 0; i < other.width; i++)
            data.set(
                    other.data,
                    index(row, col + i),
                    other.index(0, i),
                    other.height
            );

//                    .exportTo(dataPointer(row, col + i), 0, i, other.height);
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
     * Copies from the gpu stored data.
     *
     * @param length The number of elements to copy.
     * @param startRow The row in the matrix to begin copying from.
     * @param startCol The column in the matrix to begin copying from.
     * @return An array containing elements retrieved from gpu storage.
     */
    private double[] exportToArray(int startRow, int startCol, int length) {
        return data.get(index(startRow, startCol), length);

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
        return exportToArray(row, column, 1)[0];

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
        return new Dimension(endColumn - startColumn + 1, endRow - startRow + 1);
    }

    /**
     * Does some basic checks on the validity of the subMatrix parameters. Throw
     * exceptions if there are any problems.
     *
     * @param startRow
     * @param endRow
     * @param startColumn
     * @param endColumn
     * @throws OutOfRangeException
     * @throws NumberIsTooSmallException
     */
    private void checkSubMatrixParameters(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkRowCol(endRow, endColumn);
        checkRowCol(startRow, startColumn);
        if (startColumn >= endColumn)
            throw new NumberIsTooSmallException(endColumn, startColumn, true);
        if (startRow >= endRow)
            throw new NumberIsTooSmallException(endRow, startRow, true);

    }

    @Override
    public void copySubMatrix(int startRow, int endRow, int startColumn, int endColumn, double[][] destination) throws OutOfRangeException, NumberIsTooSmallException, MatrixDimensionMismatchException {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);
        if (destination.length != height || destination[0].length != width)
            throw new MatrixDimensionMismatchException(destination.length, destination[0].length, height, width);

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        for (int j = 0; j < dim.width; j++)
            destination[j] = exportToArray(startRow, startColumn + j, dim.height);
    }

    @Override
    public Matrix getSubMatrix(int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        checkSubMatrixParameters(startRow, endRow, startColumn, endColumn);

        Dimension dim = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        Matrix sub = new Matrix(handle, dim.height, dim.width);

        for (int j = 0; j < dim.width; j++)
            sub.data.set(
                    data,
                    sub.index(0, j),
                    index(startRow, j + startColumn),
                    sub.height
            );

        return sub;
    }

    /**
     * If the row is outside of this matrix, an exception is thrown.
     *
     * @param row
     * @throws OutOfRangeException
     */
    private void checkRow(int row) throws OutOfRangeException {
        if (row < 0 || row >= height)
            throw new OutOfRangeException(row, 0, height);
    }

    /**
     * If the column is outside of this matrix, an exception is thrown.
     *
     * @param col
     * @throws OutOfRangeException
     */
    private void checkCol(int col) {
        if (col < 0 || col >= width)
            throw new OutOfRangeException(col, 0, width);
    }

    /**
     * If either the row or column are out of range, an exception is thrown.
     *
     * @param row
     * @param col
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

    @Override
    public Matrix getSubMatrix(int[] selectedRows, int[] selectedColumns) throws NullArgumentException, NoDataException, OutOfRangeException {

        checkForNull(selectedRows, selectedColumns);
        if (selectedColumns.length == 0 || selectedRows.length == 0)
            throw new NoDataException();

        Matrix subMat = new Matrix(selectedRows.length, selectedColumns.length);

        int toInd = 0;

        for (int fromColInd : selectedColumns)
            for (int fromRowInd : selectedRows) {
                checkRowCol(fromRowInd, fromColInd);

                subMat.data.set(data, toInd, index(fromRowInd, fromColInd), 1);
            }

        return subMat;
    }

    @Override
    public void setSubMatrix(double[][] subMatrix, int row, int column) {
        Matrix gpuSubMatrix = new Matrix(subMatrix);
        this.insert(gpuSubMatrix, row, column);
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
     * @return
     */
    public boolean equals(Matrix object) {
        return equals(object, 1e-10);
    }

    /**
     * Checks if the two methods are equal to within an epsilon margin of error.
     *
     * @param object A matrix that might be equal to this one.
     * @param epsilon The acceptable margin of error.
     * @return True if the matrices are very close to one another, false
     * otherwise.
     */
    public boolean equals(Matrix object, double epsilon) {//TODO:This method needs to be double checked.

        return java.lang.Math.abs(data.dot(object.data)) < epsilon;
    }

    @Override
    public Matrix createMatrix(int rowDimension, int columnDimension) throws NotStrictlyPositiveException {
        if (rowDimension <= 0 || columnDimension <= 0)
            throw new NotStrictlyPositiveException(java.lang.Math.min(rowDimension, columnDimension));
        return new Matrix(rowDimension, columnDimension);
    }

    @Override
    public Matrix copy() {
        return new Matrix(data.copy(), height, width);

    }

    public static void main(String[] args) {

        Matrix matrixA = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix matrixB = new Matrix(new double[][]{{7, 8}, {9, 10}});

        // Print the original matrices
        System.out.println("Matrix A:");
        System.out.println(matrixA);
        System.out.println();

        System.out.println("Matrix B:");
        System.out.println(matrixB);
        System.out.println();

        matrixA.insert(matrixB, 1, 0);

        System.out.println("Matrix A:");
        System.out.println(matrixA);
        System.out.println();

    }

    @Override
    public int getRowDimension() {
        return height;
    }

    @Override
    public int getColumnDimension() {
        return width;
    }

    @Override
    public void setEntry(int row, int column, double value) {
        data.set(index(row, column), value);
    }

    @Override
    public Matrix getRowMatrix(int row) throws OutOfRangeException {
        if (row < 0 || row >= height)
            throw new OutOfRangeException(row, 0, height);

        Matrix rowMatrix = new Matrix(1, width);

        for (int i = 0; i < width; i++)
            rowMatrix.data.set(data, i, index(row, i), 1);

        return rowMatrix;
    }

    @Override
    public double[] getRow(int row) throws OutOfRangeException {
        if (row < 0 || row >= height)
            throw new OutOfRangeException(row, 0, height);

        double[] rowArray = new double[width];

        for (int i = 0; i < rowArray.length; i++)
            data.get(rowArray, i, index(row, i), 1);

        return rowArray;
    }

    @Override
    public RealVector getRowVector(int row) throws OutOfRangeException {//TODO: create a GPU vector class.

        return new ArrayRealVector(getRow(row));
    }

    @Override
    public double[] getColumn(int column) throws OutOfRangeException {
        if (column >= width || column < 0)
            throw new OutOfRangeException(column, 0, width);
        return exportToArray(0, column, height);
    }

    @Override
    public Matrix getColumnMatrix(int column) throws OutOfRangeException {
        return new Matrix(
                data.subArray(index(0, column), height).copy(),
                1,
                height
        );

    }

    @Override
    public RealVector getColumnVector(int column) throws OutOfRangeException {
        return new ArrayRealVector(exportToArray(0, column, height));
    }

    @Override
    public double[][] getData() {
        double[][] getData = new double[height][width];
        Arrays.setAll(getData, i -> getColumn(i));
        return getData;
    }

    @Override
    public double getTrace() throws NonSquareMatrixException {
        if (!isSquare()) throw new NonSquareMatrixException(width, height);
        double[] diagnal = new double[Math.min(height, width)];
        for (int i = 0; i < diagnal.length; i++)
            data.get(diagnal, i, index(i, i), 1);
        return Arrays.stream(diagnal).sum();
    }

    @Override
    public int hashCode() {
        return new Array2DRowRealMatrix(getData()).hashCode();
    }

    @Override
    public double[] operate(double[] v) throws DimensionMismatchException {
        if (width != v.length)
            throw new DimensionMismatchException(v.length, width);

        return DArray.emptyArray(height)
                .multMatVec(false, height, width, 1, data, new DArray(v), 1)
                .get();
    }

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
    public static Matrix identity(int n) {
        double[] ident = new double[n * n];
        for (int i = 0; i < ident.length; i += n + 1)
            ident[i] = 1;
        return new Matrix(ident, n, n);
    }

    @Override
    public Matrix power(int p) throws NotPositiveException, NonSquareMatrixException {
        if (p < 0) throw new NotPositiveException(p);
        if (!isSquare()) throw new NonSquareMatrixException(width, height);

        if (p == 0) return identity(width);

        if (p % 2 == 0) {
            Matrix halfPow = power(p / 2);
            return halfPow.multiply(halfPow);
        } else return multiply(power(p - 1));
    }

    @Override
    public void setColumn(int column, double[] array) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkCol(column);
        if (array.length != height)
            throw new MatrixDimensionMismatchException(0, array.length, 0, height);

        data.set(array, index(0, column));
    }

    /**
     * Sets the specified column of this matrix to the entries of the specified
     * column matrix.
     *
     * @param column Column to be set.
     * @param matrix Column matrix to be copied (must have one column and the
     * same number of rows as the instance).
     * @throws OutOfRangeException if the specified column index is invalid.
     * @throws MatrixDimensionMismatchException if the column dimension of the
     * matrix is not 1, or the row dimensions of this and matrix do not match.y
     */
    public void setColumnMatrix(int column, Matrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkCol(column);
        if (matrix.height != height || matrix.width != 1)
            throw new MatrixDimensionMismatchException(matrix.width, matrix.height, 1, height);

        data.set(matrix.data, index(0, column), 0, height);
    }

    @Override
    public void setColumnMatrix(int column, RealMatrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        setColumn(column, matrix.getColumn(0));
    }

    @Override
    public void setColumnVector(int column, RealVector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        setColumn(column, vector.toArray());
    }

    @Override
    public void setRow(int row, double[] array) throws OutOfRangeException, MatrixDimensionMismatchException {
        checkRow(row);
        if (array.length != width)
            throw new MatrixDimensionMismatchException(array.length, 0, width, 0);

        for (int i = 0; i < array.length; i++)
            data.set(array, index(row, i), i, 1);
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
        for (int i = 0; i < width; i++)
            data.set(rowMatrix.data, index(row, i), i, 1);
    }

    @Override
    public void setRowMatrix(int row, RealMatrix matrix) throws OutOfRangeException, MatrixDimensionMismatchException {
        setRow(row, matrix.getRow(0));
    }

    @Override
    public void setRowVector(int row, RealVector vector) throws OutOfRangeException, MatrixDimensionMismatchException {
        setRow(row, vector.toArray());
    }

    @Override
    public Matrix transpose() {
        Matrix transpose = new Matrix(width, height);

        cublasHandle handle = new cublasHandle();
        JCublas2.cublasCreate(handle);

        transpose.data.MatrixAddWithTranspose(
                handle, true, false, height, width, 1, data, 0, null, width
        );

        JCublas2.cublasDestroy(handle);

        return transpose;
    }

    @Override
    public double walkInColumnOrder(RealMatrixChangingVisitor visitor) {
        double[] matrix = data.get();

        Arrays.setAll(matrix, i -> visitor.visit(rowIndex(i), columnIndex(i), matrix[i]));

        data.set(matrix);

        return visitor.end();
    }

    @Override
    public double walkInColumnOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) {

        Dimension sub = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        double[] matrix = new double[sub.width * sub.height];

        for (int toCol = 0; toCol <= sub.width; toCol++)
            data.get(matrix, sub.height * toCol, index(0, toCol + startColumn), height);

        Arrays.setAll(matrix, i -> visitor.visit(i % sub.height + startRow, i / sub.height + startColumn, matrix[i]));

        for (int col = startColumn; col <= endColumn; col++)
            data.set(matrix, index(startRow, col), col - startColumn, sub.height);

        return visitor.end();
    }

    @Override
    public double walkInOptimizedOrder(RealMatrixChangingVisitor visitor) {

        return walkInColumnOrder(visitor);
    }

    @Override
    public double walkInOptimizedOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) {

        return walkInColumnOrder(visitor, startRow, endRow, startColumn, endColumn);
    }

    @Override
    public double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor) {
        return walkInColumnOrder(visitor);
    }

    @Override
    public double walkInRowOrder(RealMatrixChangingVisitor visitor) {
        throw new UnsupportedOperationException("Don't walk this matrix in row order.  It's a column major matrix.");
    }

    @Override
    public double walkInRowOrder(RealMatrixChangingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        throw new UnsupportedOperationException("Don't walk this matrix in row order.  It's a column major matrix.");
    }

    @Override
    public double walkInColumnOrder(RealMatrixPreservingVisitor visitor) {
        double[] matrix = exportToArray(0, 0, size());

        for (int i = 0; i < matrix.length; i++)
            visitor.visit(i / width, i % width, matrix[i]);

        return visitor.end();
    }

    @Override
    public double walkInColumnOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        Dimension sub = subMatrixDimensions(startRow, endRow, startColumn, endColumn);

        double[] matrix = new double[sub.width * sub.height];

        for (int toCol = 0; toCol <= sub.width; toCol++)
            data.get(matrix, toCol * sub.height, index(0, toCol + startColumn), sub.height);

        for (int i = 0; i < matrix.length; i++)
            visitor.visit(i % sub.height + startRow, i / sub.height + startColumn, matrix[i]);

        return visitor.end();
    }

    @Override
    public double walkInOptimizedOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        return walkInColumnOrder(visitor);
    }

    @Override
    public double walkInRowOrder(RealMatrixPreservingVisitor visitor) {
        throw new UnsupportedOperationException("Don't walk this matrix in row order.  It's a column major matrix.");
    }

    @Override
    public double walkInRowOrder(RealMatrixPreservingVisitor visitor, int startRow, int endRow, int startColumn, int endColumn) throws OutOfRangeException, NumberIsTooSmallException {
        throw new UnsupportedOperationException("Don't walk this matrix in row order.  It's a column major matrix.");
    }

    @Override
    public double getFrobeniusNorm() {
        return Math.sqrt(data.dot(data));
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

}
