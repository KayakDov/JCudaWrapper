package algebra;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.exception.NotPositiveException;
import processSupport.Handle;
import storage.DArray;
import storage.DSingleton;

/**
 * The {@code Vector} class extends {@code RealVector} and represents a vector
 * stored on the GPU. It relies on the {@code DArray} class for data storage and
 * the {@code Handle} class for JCublas operations.
 */
public class Vector extends RealVector implements AutoCloseable {

    final DArray data;  // Underlying array for GPU-based operations
    private final Handle handle; // JCublas handle for GPU operations
    final int inc;

    /**
     * Constructs a new {@code Vector} from an existing data pointer on the GPU.
     *
     * @param data The {@code DArray} storing the vector on the GPU.
     * @param inc The increment between elements of the data that make of this
     * vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(DArray data, int inc, Handle handle) {
        this.data = data;
        this.handle = handle;
        this.inc = inc;
    }

    /**
     * Constructs a new {@code Vector} from a 1D array.
     *
     * @param array The array storing the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, double... array) {
        this(new DArray(handle, array), 1, handle);
    }

    /**
     * Constructs a new empty {@code Vector} of specified length.
     *
     * @param length The length of the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(int length, Handle handle) {
        this(DArray.empty(length), 1, handle);
    }

    /**
     * Retrieves the element at the specified index.
     *
     * @param index The index of the element.
     * @return The value of the element at the specified index.
     * @throws OutOfRangeException if the index is out of range.
     */
    @Override
    public double getEntry(int index) throws OutOfRangeException {
        checkIndex(index);
        return data.get(index * inc).getVal();
    }

    /**
     * Sets the element at the specified index to the given value.
     *
     * @param index The index to set.
     * @param value The value to assign.
     * @throws OutOfRangeException if the index is out of range.
     */
    @Override
    public void setEntry(int index, double value) throws OutOfRangeException {
        checkIndex(index);
        data.set(handle, index * inc, value);
    }

    /**
     * Returns the dimension (length) of the vector.
     *
     * @return The length of the vector.
     */
    @Override
    public int getDimension() {
        return Math.ceilDiv(data.length , inc);
    }

    /**
     * Adds another vector times a scalar to this vector, changing this vector.
     *
     * @param mult A scalar to be multiplied by @code{v} before adding it to
     * this vector.
     * @param v The vector to be added to this vector.
     * @return This vector.
     */
    public Vector addToMe(double mult, Vector v) {
        checkVectorLength(v);

        data.addToMe(handle, mult, v.data, v.inc, inc);
        return this;
    }

    /**
     * Multiplies this vector by a scalar changing the elements in this vector.
     *
     * @param scalar The scalar to be multiplied by this vector.
     * @return This vector.
     */
    public Vector multiplyMe(double scalar) {
        data.multMe(handle, scalar, inc);
        return this;
    }

    /**
     * Sets all the values in this vector to that of the scalar.
     *
     * @param scalar The new value to fill this vector.
     * @return This vector.
     */
    public Vector fill(double scalar) {
        if (scalar == 0 && inc == 1) data.fill0(handle);
        else data.fill(handle, scalar, inc);
        return this;
    }

    /**
     * Adds this vector to another vector element-wise.
     *
     * @param v The vector to add.
     * @return A new vector that is the sum of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    @Override
    public Vector add(RealVector v) throws DimensionMismatchException {

        try (Vector other = new Vector(handle, v.toArray())) {
            return add(other);
        }
    }

    /**
     * Adds this vector to another vector element-wise.
     *
     * @param v The vector to add.
     * @return A new vector that is the sum of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector add(Vector v) throws DimensionMismatchException {

        return copy().addToMe(1, v);
    }

    /**
     * Multiplies each element of this vector by a scalar.
     *
     * @param d The scalar multiplier.
     * @return A new vector that is this vector scaled by {@code d}.
     */
    @Override
    public Vector mapMultiply(double d) {

        return copy().multiplyMe(d);
    }

    /**
     * Computes the dot product of this vector with another vector.
     *
     * @param v The other vector to compute the dot product with.
     * @return The dot product of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    @Override
    public double dotProduct(RealVector v) throws DimensionMismatchException {

        try (Vector temp = new Vector(handle, v.toArray())) {
            return dotProduct(temp);
        }

    }

    /**
     * Computes the dot product of this vector with another vector.
     *
     * @param v The other vector to compute the dot product with.
     * @return The dot product of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public double dotProduct(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);

        return data.dot(handle, v.data, v.inc, inc);
    }

    /**
     * Subtracts another vector from this vector element-wise.
     *
     * @param v The vector to subtract.
     * @return A new vector that is the difference of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    @Override
    public RealVector subtract(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return subtract(temp);
        }
    }

    /**
     * Subtracts another vector from this vector element-wise.
     *
     * @param v The vector to subtract.
     * @return A new vector that is the difference of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector subtract(Vector v) throws DimensionMismatchException {

        return copy().addToMe(-1, v);
    }

    /**
     * Creates a deep copy of this vector.
     *
     * @return A new {@code Vector} that is a deep copy of this vector.
     */
    @Override
    public Vector copy() {
        if (inc == 1) return new Vector(data.copy(handle), inc, handle);
        return new Vector(getDimension(), handle).fill(0).add(this);
    }

    /**
     * Computes the element-wise product of this vector and another vector.
     *
     * @param v The other vector.
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    @Override
    public Vector ebeMultiply(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return ebeMultiply(temp);
        }
    }

    /**
     * Computes the element-wise product of this vector and another vector.
     *
     * @param v The other vector.
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeMultiply(Vector v) {
        checkVectorLength(v);

        Vector result = new Vector(data.length, handle);

        result.data.multSymBandMatVec(handle, true, v.getDimension(), 0, 1, v.data, 1, data, inc, 1, 1);

        return result;
    }

    /**
     * Computes the element-wise division of this vector by another vector.
     *
     * @param v The vector to divide by.
     * @return A new vector containing the element-wise division of this vector
     * by {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    @Override
    public Vector ebeDivide(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return ebeDivide(temp);
        }
    }

    /**
     * Computes the element-wise division of this vector by another vector.
     *
     * @param v The vector to divide by.
     * @return A new vector containing the element-wise division of this vector
     * by {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeDivide(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);

        Vector inverse = new Vector(v.getDimension(), handle).fill(1);

        inverse.data.solveTriangularBandedSystem(handle, true, false, false,
                v.getDimension(), 0, v.data, 1, 1);

        return ebeMultiply(inverse);

    }

    /**
     * Computes the Euclidean norm (2-norm) of the vector.
     *
     * @return The Euclidean norm of the vector.
     */
    @Override
    public double getNorm() {
        return data.norm(handle).getVal();
    }

    /**
     * Checks if the index is within the valid range.
     *
     * @param index The index to check.
     * @throws OutOfRangeException if the index is out of range.
     */
    @Override
    public void checkIndex(int index) {
        if (index < 0 || index >= getDimension())
            throw new OutOfRangeException(index, 0, getDimension() - 1);
    }

    /**
     * Checks if the dimensions of this vector match the dimensions of the given
     * vector.
     *
     * @param v The vector to compare dimensions with.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    private void checkVectorLength(RealVector v) {
        if (v.getDimension() != getDimension())
            throw new DimensionMismatchException(v.getDimension(), getDimension());
    }

    /**
     * A new vector that is the concatenation of this vector and given one. This
     * vector does not change.
     *
     * @param rv The vector to be concatenated to this one.
     * @return A The concatenation of the two vectors.
     */
    @Override
    public Vector append(RealVector rv) {
        try (Vector temp = new Vector(handle, rv.toArray())) {
            return append(temp);
        }
    }

    /**
     * A new vector that is the concatenation of this vector and given one. This
     * vector does not change.
     *
     * @param rv The vector to be concatenated to this one.
     * @return A The concatenation of the two vectors.
     */
    public Vector append(Vector rv) {
        DArray append = DArray.empty(getDimension() + rv.getDimension());
        append.set(handle, data, 0, 0, 1, inc, getDimension());
        append.set(handle, rv.data, getDimension(), 0, 1, rv.inc, rv.getDimension());
        return new Vector(data, 1, handle);
    }

    /**
     * A new vector that is the concatenation of this vector and given one. This
     * vector does not change.
     *
     * @param d The element to be concatenated to this vector.
     * @return A The concatenation of the two vectors.
     */
    @Override
    public Vector append(double d) {
        return append(new Vector(new DSingleton(d, handle), 1, handle));
    }

    /**
     * Retrieves a subvector from this vector starting at the specified index
     * and of the specified length.
     *
     * The method creates a new {@link Vector} that shares the underlying data
     * with the original vector, but starts from the specified index and
     * includes only the specified number of elements.
     *
     * @param index The starting index of the subvector in the original vector.
     * This is scaled by the increment (inc).
     * @param length The number of elements in the subvector to be extracted.
     * @return A new {@link Vector} containing the elements from the specified
     * subrange.
     * @throws NotPositiveException If the length is not positive.
     * @throws OutOfRangeException If the index is out of the valid range.
     */
    @Override
    public Vector getSubVector(int index, int length) throws NotPositiveException, OutOfRangeException {
        return new Vector(data.subArray(index * inc, length), inc, handle);
    }

    /**
     * Sets a subvector of this vector starting at the specified index using the
     * elements of the provided {@link RealVector}.
     *
     * This method modifies this vector by copying the elements of the given
     * {@link RealVector} into this vector starting at the position defined by
     * the index, with the length of the subvector determined by the dimension
     * of the given real vector.
     *
     * @param i The starting index where the subvector will be set, scaled by
     * the increment (inc).
     * @param rv The {@link RealVector} whose elements will be copied into this
     * vector.
     * @throws OutOfRangeException If the index is out of range.
     */
    @Override
    public void setSubVector(int i, RealVector rv) throws OutOfRangeException {
        data.set(handle, rv.toArray(), i * inc, 0, rv.getDimension());
    }

    /**
     * Sets a subvector of this vector starting at the specified index using the
     * elements of the provided {@link Vector}.
     *
     * This method modifies this vector by copying the elements of the given
     * {@link Vector} into this vector starting at the position defined by the
     * index, with the length of the subvector determined by the dimension of
     * the given vector.
     *
     * @param i The starting index where the subvector will be set, scaled by
     * the increment (inc).
     * @param rv The {@link Vector} whose elements will be copied into this
     * vector.
     * @throws OutOfRangeException If the index is out of range.
     */
    public void setSubVector(int i, Vector rv) throws OutOfRangeException {
        data.set(handle, rv.data, i * inc, 0, rv.getDimension());
    }

    /**
     * Checks if this vector contains any NaN (Not-a-Number) values.
     *
     * This method determines if the dot product of this vector with itself
     * results in NaN. If the dot product is NaN, this indicates that the vector
     * contains at least one NaN element.
     *
     * @return {@code true} if this vector contains NaN values, {@code false}
     * otherwise.
     */
    @Override
    public boolean isNaN() {
        return Double.isNaN(dotProduct(this));
    }

    /**
     * Checks if this vector contains any infinite values.
     *
     * This method determines if the dot product of this vector with itself
     * results in an infinite value. If the dot product is infinite, this
     * indicates that the vector contains at least one infinite element.
     *
     * @return {@code true} if this vector contains infinite values,
     * {@code false} otherwise.
     */
    @Override
    public boolean isInfinite() {
        return Double.isInfinite(dotProduct(this));
    }

    /**
     * Closes the underlying memory, so be sure it's not being used elsewhere.
     */
    @Override
    public void close() {
        if (inc != 1)
            throw new IllegalAccessError("You are cleaning data from a sub vector");
        data.close();
    }

    @Override
    public String toString() {
        return copy().data.toString();
    }

    /**
     * Compares this vector to another vector and checks if they are equal
     * within a specified tolerance.
     *
     * This method subtracts the given vector from this vector and compares the
     * norm of the resulting vector to the specified epsilon. If the norm is
     * less than epsilon, the vectors are considered equal, indicating that the
     * difference between them is smaller than the given tolerance.
     *
     * @param other The vector to compare with this vector.
     * @param epsilon The tolerance value within which the vectors are
     * considered equal. Must be a non-negative number.
     * @return true if the difference between the vectors is less than the
     * specified epsilon, false otherwise.     *
     */
    public boolean equals(Vector other, double epsilon) {
        return subtract(other).getNorm() < epsilon;
    }

    /**
     * Compares this vector to another vector and checks if they are equal
     * within a default tolerance of 1e-10.
     *
     * This method subtracts the given vector from this vector and compares the
     * norm of the resulting vector to the specified epsilon. If the norm is
     * less than epsilon, the vectors are considered equal, indicating that the
     * difference between them is smaller than the given tolerance.
     *
     * @param other The vector to compare with this vector.
     * @return true if the difference between the vectors is less than the
     * specified epsilon, false otherwise.     *
     */
    public boolean equals(Vector other) throws MathUnsupportedOperationException {
        return equals(other, 1e-10);
    }

}
