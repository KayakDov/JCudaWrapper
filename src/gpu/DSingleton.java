package gpu;

import jcuda.Pointer;

/**
 * S ingle element in the gpu.
 * @author E. Dov Neimand
 */
public class DSingleton extends DArray{
    
    /**
     * Creates a singleton by taking an element from another array.
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
        super(empty(1), 1);
    }
    /**
     * Creates a singleton from a cpu element.
     * @param d The element in the singleton.
     */
    public DSingleton(double d){
        super(d);
    }
    
    public static void main(String[] args) {
        DSingleton a = new DSingleton(5);
        
        System.out.println(a.toString());
    }
    
    
}
