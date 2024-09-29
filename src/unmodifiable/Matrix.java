package unmodifiable;

import algebra.Vector;
import java.awt.Dimension;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.NullArgumentException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.linear.MatrixDimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import resourceManagement.Handle;
import array.DArray;

/**
 * A matrix that can not me modified.
 *
 * @author E. Dov Neimand
 */
public class Matrix extends algebra.Matrix {



    /**
     * @see Matrix#Matrix(resourceManagement.Handle, storage.DArray, int, int,
     * int)
     * @param handle
     * @param array
     * @param height
     * @param width
     * @param distBetweenFirstElementOfColumns
     */
    public Matrix(Handle handle, DArray array, int height, int width, int distBetweenFirstElementOfColumns) {
        super(handle, array.unmodifiable(), height, width, distBetweenFirstElementOfColumns);
    }

}
