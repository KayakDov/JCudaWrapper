package storage;

import jcuda.Pointer;
import processSupport.Handle;

/**
 * S ingle element in the gpu.
 * @author E. Dov Neimand
 */
public class DSingleton extends DArray{
    
    /**
     * Creates a singleton by taking an element from another array.
     * Note, this is copy by reference, so changes to this singleton will
     * effect the original array.
     * @param from The array the singleton is to be taken from.
     * @param index The index in the array.
     */
    public DSingleton(DArray from, int index){
        super(from.pointer(index), 1);
    }
    
    /**
     * Creates a singleton with no assigned value.
     */
    public DSingleton(){
        super(Array.empty(1, PrimitiveType.DOUBLE), 1);
    }
    /**
     * Creates a singleton from a cpu element.
     * @param d The element in the singleton.
     * @param hand The handle.
     */
    public DSingleton(Handle hand, double d){
        super(hand, d);
    }
    
    /**
     * Gets the value in this.
     * @param hand This should be the same handle that's used to make whatever
     * results are being retrieved.  The handle is synchronized before the result
     * is returned.
     * @return The value in this singleton.
     */
    public double getVal(Handle hand){
        double[] val = get(hand);
        
        hand.synch();
        
        return val[0];
    }    
    
}
