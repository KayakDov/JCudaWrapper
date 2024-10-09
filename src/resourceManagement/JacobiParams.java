
// This code doesn't work:
//     syevjInfo params = new syevjInfo();
//     JCusolverDn.cusolverDnCreateSyevjInfo(params);
 
package resourceManagement;

import java.lang.ref.Cleaner;
import jcuda.jcusolver.JCusolverDn;
import jcuda.jcusolver.syevjInfo;

/**
 * A utility class for managing the parameters used in the Jacobi algorithm for
 * computing eigenvalues and eigenvectors of symmetric matrices in JCuda. This
 * class implements {@link AutoCloseable}, allowing for automatic resource
 * management when used in a try-with-resources statement. It handles the
 * creation and destruction of the {@link syevjInfo} structure, ensuring proper
 * cleanup of resources.
 *
 * <p>
 * Example usage:</p>
 * <pre>
 * try (JacobiParams jacobiParams = new JacobiParams()) {
 *     // Use jacobiParams.getParams() to access the parameter structure
 * }
 * </pre>
 *
 * <p>
 * This class registers the parameters structure with a {@link Cleaner} to
 * ensure that it is destroyed when the {@link close()} method is called, either
 * explicitly or implicitly.</p>
 *
 * @author E. Dov Neimand
 */
public class JacobiParams implements AutoCloseable {

    /**
     * The syevjInfo structure that holds parameters for the Jacobi algorithm.
     */
    private syevjInfo params;

    /**
     * Cleanable resource for the Jacobi parameters, used to ensure that the
     * syevjInfo structure is destroyed when no longer needed.
     */
//    private final Cleaner.Cleanable cleanableParam;

    /**
     * Constructs a new {@code JacobiParams} instance, creating the syevjInfo
     * structure necessary for the Jacobi algorithm's parameters. This structure
     * is registered for automatic cleanup.
     */
    public JacobiParams() {
    
        params = new syevjInfo(); // Correct type for Jacobi parameters

//        JCusolverDn.cusolverDnCreateSyevjInfo(params); // Create parameter structure


        // Register the syevjInfo structure for cleanup
//        cleanableParam = ResourceDealocator.register(
//                this,
//                params -> JCusolverDn.cusolverDnDestroySyevjInfo(params),
//                params
//        );
    }

    public static void main(String[] args) {
        syevjInfo params = new syevjInfo();
//        JCusolverDn.cusolverDnCreateSyevjInfo(params);
        
        
//        JCusolverDn.cusolverDnDestroySyevjInfo(params);
    }

    /**
     * Returns the syevjInfo structure that holds the parameters for the Jacobi
     * algorithm.
     *
     * @return The {@link syevjInfo} instance containing the parameters.
     */
    public syevjInfo getParams() {
        return params;
    }

    /**
     * Closes the Jacobi parameters, ensuring that the resources are cleaned up.
     * This method is automatically called when the object is used in a
     * try-with-resources statement.

     */
    @Override
    public void close() {
//        cleanableParam.clean(); // Clean up the Jacobi parameters
    }
}
