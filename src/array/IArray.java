package array;

import java.util.Arrays;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import resourceManagement.Handle;

/**
 * An array of integers.
 *
 * @author E. Dov Neimand
 */
public class IArray extends Array {

    /**
     * Constructs an empty array.
     *
     * @param p A pointer to the first element of the array.
     * @param length The length of the array.
     */
    private IArray(CUdeviceptr p, int length) {
        super(p, length, PrimitiveType.INT);
    }

    /**
     * @see DArray#empty(int)
     * @param length The length of the array.
     * @return An empty asrray.
     */
    public static IArray empty(int length) {
        return new IArray(empty(length, PrimitiveType.INT), length);
    }

    /**
     * {@inheritDoc}
     *
     * @param handle The handle.
     * @return A copy of this array.
     */
    @Override
    public IArray copy(Handle handle) {
        IArray copy = empty(length);
        copy.set(handle, pointer, length);
        return copy;
    }

    /**
     * Exports the contents of this array to the cpu array.
     *
     * @param handle The handle.
     * @param toCPUArray The cpu array into which will be copied the elements of
     * this array.
     * @param toStart The index in the array to start copying to.
     * @param fromStart The index in this array to start copying from.
     * @param length The number of elements to copy.
     */
    public void get(Handle handle, int[] toCPUArray, int toStart, int fromStart, int length) {
        super.get(handle, Pointer.to(toCPUArray), toStart, fromStart, length);
    }
    
    
    /**
     * Exports the contents of this array to the cpu.
     *
     * @param handle The handle.
     * @return The contents of this array stored in a cpu array.
     */
    public int[] get(Handle handle) {
        
        int[] toCPUArray = new int[length];
        super.get(handle, Pointer.to(toCPUArray), 0, 0, length);
        handle.synch();
        return toCPUArray;
    }

    /**
     * A pointer to a singleton array containing d.
     *
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
    static Pointer cpuPointer(int d) {
        return Pointer.to(new int[]{d});
    }

    @Override
    public String toString() {
        try (Handle hand = new Handle()) {
            int[] cpuArray = new int[length];
            get(hand, cpuArray, 0, 0, length);
            return Arrays.toString(cpuArray);
        }
    }

}
