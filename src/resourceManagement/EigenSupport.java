package resourceManagement;

import algebra.Matrix;
import algebra.Vector;
import array.DArray;
import array.DArray2d;
import array.IArray;

/**
 * The EigenSupport class provides support for computing eigenvalues and 
 * eigenvectors for a batch of matrices using the cuSolver library. 
 * It manages the resources necessary for these computations, including 
 * the workspace and information storage.
 * 
 * This class implements AutoCloseable, allowing for resource cleanup
 * through the close method.
 * 
 * @author E. Dov Neimand
 */
public class EigenSupport implements AutoCloseable {

    private JacobiParams jp;
    private DArray workspace;
    private IArray info;
    private Handle handle;

    /**
     * Constructs an EigenSupport instance, initializing the necessary 
     * parameters and creating the workspace for eigenvalue computations.
     *
     * @param handle The handle to the cuSolver context.
     * @param sampleData A sample matrix used to determine the size of the 
     *                   workspace.
     * @param sampleResultValues A vector to store the eigenvalues.
     * @param batchSize The number of matrices in the batch.
     */
    public EigenSupport(Handle handle, Matrix sampleData, Vector sampleResultValues, int batchSize) {
        handle.synch();
        System.out.println("resourceManagement.EigenSupport.<init>() Beginning eigan support");
        
        this.handle = handle;
        this.jp = new JacobiParams();
        
        System.out.println("resourceManagement.EigenSupport.<init>() checking workspace size");
        
        int workspaceSize = DArray2d.eigenWorkspaceSize(
                handle, 
                sampleData.getHeight(), 
                sampleData.dArray(), 
                sampleData.colDist, 
                sampleResultValues.dArray(), 
                batchSize, 
                jp, 
                DArray2d.Fill.FULL);
        
        System.out.println("resourceManagement.EigenSupport.<init>() workspaceSize = " + workspaceSize);
        
        workspace = DArray.empty(workspaceSize);
        
        this.info = IArray.empty(batchSize);
    }
    
    /**
     * Computes the eigenvalues and eigenvectors for the given matrix 
     * and stores the results in the provided vector.
     *
     * @param a The matrix for which to compute eigenvalues and eigenvectors.
     * @param resultValues The vector where the eigenvalues will be stored.
     */
    public void compute(Matrix a, Vector resultValues) {
        a.eigenBatch(resultValues, workspace, jp, info);
    }

    /**
     * Closes the EigenSupport instance, releasing resources associated 
     * with Jacobi parameters, workspace, info array, and the handle.
     */
    @Override
    public void close() {
        jp.close();
        workspace.close();
        info.close();
        handle.close();
    }

}
