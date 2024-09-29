
package unmodifiable;

import jcuda.driver.CUdeviceptr;
import resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class DSingleton extends DArray{
    
    /**
     * An unmodifiable copy.
     * @param p A pointer to the element
     */
    public DSingleton(CUdeviceptr p) {
        super(p, 1);
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
