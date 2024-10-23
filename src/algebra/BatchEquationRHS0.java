package algebra;

/**
 * This method represents a batch of equations Ax=b. A is a set of square
 * matrices.
 *
 * @author E. Dov Neimand
 */
public class BatchEquationRHS0 {

    private MatrixStride lhs;

    /**
     * Constructs a set of equations of the form Ax = b.
     *
     * @param a The left hand side.
     * @param b The right hand side.
     */
    public BatchEquationRHS0(MatrixStride a, Vector b) {
        this.lhs = a;
    }
    
    public void diagnolize(Vector workSpace, Vector solution){
        trianglize(workSpace);
        
        
    }
    
    /**
     * Turns this system into an upper right triangle
     * @param workSpace Should be as long as batchsize.
     */
    public void trianglize(Vector workSpace){
         for(int i = 0; i < lhs.getHeight() - 1; i++){
             Vector diagElmnt = lhs.get(i, i);
             diagElmnt.mapEBEInverse(workSpace);
             workSpace.mapEbeMultiplyToSelf(workSpace, lhs.get(i + 1, i));
             multRow(i, workSpace);
             lhs.getRowVector(i + 1).addToMe(-1, lhs.getRowVector(i));
         }
             
    }

    /**
     * Multiplies every element, from the diagonal to the right end, in each
     * subrow of this matrix by the element in the coresponding vector.
     *
     * @param row The row and column where the multiplication starts.
     * @param scalar The scalar to multiply the row by.
     */
    public void multRow(int row, Vector scalar) {//TODO: Is there a faster way to do this?
        
        for (int i = row; i < lhs.getWidth(); i++){
            Vector elementInRow = lhs.get(row, i);
            elementInRow.mapEbeMultiplyToSelf(elementInRow, scalar);
        }
    }

}
