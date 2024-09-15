package algebra;

import storage.DArray;
import java.lang.ref.Cleaner;
import java.lang.ref.ReferenceQueue;
import jcuda.Pointer;
import java.lang.ref.PhantomReference;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.MatrixDimensionMismatchException;
import org.apache.commons.math3.linear.RealVector;

/**
 *
 * @author E. Dov Neimand
 */
public class Vector extends RealVector {

    public final DArray data;

    /**
     * Creates an array from the passed data.
     *
     * @param data The data to be held in the GPU array.
     */
    public Vector(double[] data) {
        this(new DArray(data));
    }

    /**
     * Constructs this vector by copying toArray into GPU memory.
     *
     * @param rv
     */
    public Vector(RealVector rv) {
        this(rv.toArray());
    }

    /**
     * Uses p as data. Changes to data will change p.
     *
     * @param array The pointer whose contents are to be copied.
     * @param length The number of elements to be copied.
     */
    Vector(DArray array) {
        this.data = array;
    }

    @Override
    public Vector copy() {
        return new Vector(data.copy());
    }

    /**
     * Adds this array and v to create a new array.
     *
     * @param v The vector to be added to this one.
     * @return The sum of this array and v.
     * @throws DimensionMismatchException
     */
    public Vector add(Vector v) throws DimensionMismatchException {
        Vector add = copy();
        add.data.addToMe(1, v.data);
        return add;
    }

    /**
     * This vector minus the other vector.
     *
     * @param v The vector to be subtracted from this one.
     * @return The resulting difference, in a new vector.
     */
    public Vector subtract(Vector v) {
        return new Vector(data.copy().addToMe(-1, v.data));
    }

    /**
     * Creates a new vector that is the product of this vector and the given
     * scalar.
     *
     * @param scalar
     * @return A new vector that is the product of this vector and the given
     * scalar.
     */
    public Vector multiply(double scalar) {
        Vector mult = copy();
        mult.data.multMe(scalar);
        return mult;

    }

    /**
     * An empty vector of the given size.
     * @param size The number of dimensions in the new vector.
     * @return An empty vector.  It may have gibberish values.
     */
    public static Vector empty(int size) {
        return new Vector(DArray.empty(size));
    }

    
    public double dot(Vector v) throws DimensionMismatchException {
        if(v.getDimension() != getDimension()) throw new DimensionMismatchException(v.getDimension(), getDimension());
        
        return data.dot(v.data);
        
    }
    
    

}
