package unmodifiable;

import resourceManagement.Handle;

/**
 * A vector that can not be modified.
 *
 * @author E. Dov Neimand
 */
public class Vector extends algebra.Vector {

    /**
     * Makes a matrix unmodifiable.
     *
     * @param handle The handle of the underlying modifiable vector.
     * @param data The data of the underlying modifiable vector.
     * @param inc the increment of the underlying modifiable vector.
     */
    public Vector(Handle handle, array.DArray data, int inc) {
        super(handle, data.unmodifiable(), inc);
    }
}
