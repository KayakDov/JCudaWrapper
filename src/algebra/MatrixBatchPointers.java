package algebra;

import java.awt.Point;
import java.util.function.Consumer;

import org.apache.commons.math3.exception.DimensionMismatchException;
import resourceManagement.Handle;
import array.DArray2d;

/**
 * A class representing a batch of matrices stored in GPU memory, allowing
 * efficient operations such as batched matrix-matrix multiplications.
 *
 * <p>
 * This class provides a wrapper around the underlying {@link DArray2d} data
 * structure and offers batch matrix operations that utilize the GPU for high
 * performance.
 * </p>
 *
 * <p>
 * Each matrix in the batch is expected to have the same dimensions, and the
 * column distance (stride) between matrix elements can be controlled via the
 * {@code colDist} parameter.
 * </p>
 *
 * <p>
 * The class supports optional transposition of matrices during operations via
 * the {@code transposeForOperations} flag.
 * </p>
 *
 * @author E. Dov Neimand
 */
public class MatrixBatchPointers implements AutoCloseable {

    /**
     * The number of rows (height) of each matrix in the batch.
     */
    private final int height;

    /**
     * The number of columns (width) of each matrix in the batch.
     */
    private final int width;

    /**
     * The number of elements between the first element of each column of each
     * matrix (column stride or leading dimension).
     */
    private final int colDist;

    /**
     * The underlying data structure that stores the batch of matrices in GPU
     * memory.
     */
    private final DArray2d arrays;

    /**
     * Indicates whether the matrices should be transposed during operations. If
     * true, matrices are transposed before being used in operations.
     */
    public boolean transposeForOperations;

    /**
     * Constructs a batch of matrices.
     *
     * @param height The height (number of rows) of each matrix.
     * @param width The width (number of columns) of each matrix.
     * @param colDist The number of elements between the first element of each
     * column (column stride or leading dimension).
     * @param arrays The underlying data arrays that store the batch of
     * matrices.
     */
    public MatrixBatchPointers(int height, int width, int colDist, DArray2d arrays) {
        this.height = height;
        this.width = width;
        this.colDist = colDist;
        this.arrays = arrays;
    }

    /**
     * Creates a batch of matrices from the sub matrices of contains.
     *
     * @param contains The Matrix containing all the sub matrices that will go
     * into the batch.
     * @param step Accepts a pair of indices in the matrix and moves that pair
     * to the upper left corner of the next matrix to appear in the batch.
     * @param height The height of each sub matrix.
     * @param width The width of each sub matrix.
     * @param batchSize The number of sub matrices.
     */
    public MatrixBatchPointers(Matrix contains, Consumer<Point> step, int height, int width, int batchSize) {

        this(
                height,
                width,
                contains.colDist,
                getDarray(contains, step, height, width, batchSize)
        );
    }

    /**
     * Creates a batch of matrices from the sub matrices of contains.
     *
     * @param contains The Matrix containing all the sub matrices that will go
     * into the batch.
     * @param downStride How far down to go for the next matrix.
     * @param rightStride Once a column is complete, go this is the distance to
     * the next column.
     * @param height The height of each sub matrix.
     * @param width The width of each sub matrix.
     */
    public MatrixBatchPointers(Matrix contains, int downStride, int rightStride, int height, int width) {

        this(contains, p -> {
            p.y += downStride;
            if (p.y >= contains.getHeight()) {
                p.y = 0;
                p.x += rightStride;
            }
        },
                height, width,
                (contains.getHeight() / downStride) * (contains.getWidth() / rightStride));
    }

    /**
     * Extracts a DArray2d to serve as the underlying data for a matrixBatch.
     *
     * @param contains The Matrix containing all the sub matrices that will go
     * into the batch.
     * @param step Accepts a pair of indices in the matrix and moves that pair
     * to the upper left corner of the next matrix to appear in the batch.
     * @param height The height of each sub matrix.
     * @param width The width of each sub matrix.
     * @param batchSize The number of sub matrices.
     * @return The data for each sub matrix.
     */
    private static DArray2d getDarray(Matrix contains, Consumer<Point> step, int height, int width, int batchSize) {
        array.DArray[] arrays = new array.DArray[batchSize];

        Point p = new Point(0, 0);

        for (int i = 0; i < batchSize; i++) {
            step.accept(p);
            arrays[i] = contains.getSubMatrix(p.y, p.y + height, p.x, p.x + width).dArray();
        }
        return new DArray2d(contains.getHandle(), arrays);
    }

    /**
     * Adds the result of a matrix-matrix multiplication to this matrix batch.
     *
     * <p>
     * The multiplication is computed as:
     * <pre>
     * this = timesAB * (A * B) + timesThis * this
     * </pre>
     * </p>
     *
     * @param handle The cuBLAS handle used to manage GPU operations.
     * @param timesAB The scalar value to multiply the result of the
     * multiplication of A and B.
     * @param a The left-hand side matrix batch A.
     * @param b The right-hand side matrix batch B.
     * @param timesThis The scalar value to multiply this matrix batch before
     * adding the result.
     * @return This matrix batch (modified in place).
     * @throws DimensionMismatchException If the dimensions of matrix batch A
     * and matrix batch B are not compatible for multiplication.
     * @throws ArrayIndexOutOfBoundsException If the sizes of the matrix batches
     * A, B, and this batch do not match.
     */
    public MatrixBatchPointers addToMeMatMatMult(Handle handle, double timesAB, MatrixBatchPointers a, MatrixBatchPointers b, double timesThis) {
        // Ensure the dimensions are compatible for matrix multiplication
        if (a.width != b.height) {
            throw new DimensionMismatchException(a.width, b.height);
        }
        // Ensure all batches have the same number of matrices
        if (a.arrays.length != b.arrays.length || a.arrays.length != arrays.length) {
            throw new ArrayIndexOutOfBoundsException("Batches are not the same size. "
                    + "A batch = " + a.arrays.length + ", B batch = " + b.arrays.length
                    + ", this batch = " + arrays.length);
        }

        // Perform the batched matrix-matrix multiplication
        arrays.multMatMatBatched(handle,
                a.transposeForOperations, b.transposeForOperations,
                a.height, a.width, b.width,
                timesAB,
                a.arrays, a.colDist,
                b.arrays, b.colDist,
                timesThis, colDist,
                arrays.length
        );
        return this;
    }

    /**
     * Returns the height (number of rows) of each matrix in the batch.
     *
     * @return The height of each matrix in the batch.
     */
    public int getHeight() {
        return height;
    }

    /**
     * Returns the width (number of columns) of each matrix in the batch.
     *
     * @return The width of each matrix in the batch.
     */
    public int getWidth() {
        return width;
    }

    /**
     * Returns the column distance (leading dimension) of each matrix in the
     * batch.
     *
     * @return The column distance of each matrix in the batch.
     */
    public int getColDist() {
        return colDist;
    }

    /**
     * Returns the number of matrices in the batch.
     *
     * @return The number of matrices in the batch.
     */
    public int getBatchSize() {
        return arrays.length;
    }

    /**
     * Checks if the batch of matrices is empty.
     *
     * @return True if the batch contains no matrices, false otherwise.
     */
    public boolean isEmpty() {
        return arrays.length == 0;
    }

    /**
     * Transposes each matrix in the batch for future operations.
     *
     * <p>
     * This method sets the {@code transposeForOperations} flag to true, which
     * indicates that matrices should be transposed when used in future
     * operations.
     * </p>
     */
    public void transpose() {
        transposeForOperations = !transposeForOperations;
    }

    /**
     * Resets the transposition flag, so matrices are not transposed in future
     * operations.
     */
    public void resetTranspose() {
        this.transposeForOperations = false;
    }

    /**
     * Copies this batch but still references same location in gpu memory.
     *
     * @return A shallow copy of this batch.
     */
    public MatrixBatchPointers shallowCopy() {
        MatrixBatchPointers copy = new MatrixBatchPointers(height, width, colDist, arrays);
        copy.transposeForOperations = transposeForOperations;
        return copy;
    }

    /**
     * closes the underlying array of pointers to data. This should only be
     * called if the resource is not being used by any other objects. This
     * method does not close the data being pointed to by the elements of the
     * array of pointers.
     */
    @Override
    public void close() {
        arrays.close();
    }
    
}
