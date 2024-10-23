package algebra;

import array.DBatchArray;
import resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class VectorStride extends Vector {

    public final DBatchArray data;
    public final int subVecLength;

    /**
     * The constructor.
     * @param handle
     * @param data 
     * @param subVecLength The length of each subvector.
     * @param inc The increment of each subvector.
     */
    public VectorStride(Handle handle, DBatchArray data, int subVecLength, int inc) {
        super(handle, data, inc);
        this.data = data;
        this.subVecLength = subVecLength;
    }
    
    /**
     *
     * @param handle
     * @param data The underlying data.
     * @param stride The number of elements in the data that are between the beginnings of the subvectors.
     * @param subVecLength The number of elements in a sub vector.
     
     */
    public VectorStride(Handle handle, Vector data, int stride, int subVecLength) {
        this(handle, data.dArray().getStrided(stride*data.inc), subVecLength, data.inc);
    }

}
