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
     */
    public DSingleton(double d, Handle hand){
        super(hand, d);
    }
    
    /**
     * Gets the value in this.
     * @return The value in this singleton.
     */
    public double getVal(){
        double[] get;
        try(Handle hand = new Handle()){
             get = get(hand);
        }
        return get[0];
    }    
    
}
