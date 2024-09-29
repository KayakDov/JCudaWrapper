package resourceManagement;

import java.lang.ref.Cleaner;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

/**
 * A utility class for managing a CUDA CUBLAS handle and a corresponding CUDA stream.
 * This class implements {@link AutoCloseable}, allowing for automatic resource management
 * when used in a try-with-resources statement. It handles the creation, association, 
 * and destruction of a {@link cublasHandle} and {@link cudaStream_t}, ensuring that 
 * resources are properly cleaned up.
 * 
 
 * Example usage:
 * <pre>
 * try (Handle handle = new Handle()) {
 *     // Use handle.get() to perform operations with the handle
 * }
 * </pre>
 
 * 
 * <p>
 * This class registers the handle and stream with a {@link Cleaner} to ensure that 
 * both the CUBLAS handle and the CUDA stream are destroyed when the {@link close()} method 
 * is called, either explicitly or implicitly.
 * </p>
 * 
 * @author edov
 */
public class Handle implements AutoCloseable {

    /**
     * Cleanable resource for the CUBLAS handle, used to ensure the handle is destroyed when no longer needed.
     */
    private final Cleaner.Cleanable cleanableHandle;

    /**
     * Cleanable resource for the CUDA stream, used to ensure the stream is destroyed when no longer needed.
     */
    private final Cleaner.Cleanable cleanableStream;

    /**
     * The CUBLAS handle for managing CUBLAS operations.
     */
    private cublasHandle handle;

    /**
     * The CUDA stream associated with the CUBLAS handle.
     */
    private cudaStream_t stream;

    /**
     * Constructs a new {@code Handle}, creating a CUBLAS handle and associating it 
     * with a new CUDA stream. Both the handle and the stream are registered for automatic cleanup.
     */
    public Handle() {
        // Initialize and create the CUBLAS handle
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        
        // Initialize and create the CUDA stream
        stream = new cudaStream_t();
        JCuda.cudaStreamCreate(stream);

        // Associate the CUBLAS handle with the CUDA stream
        JCublas2.cublasSetStream(handle, stream);
        
        // Register the handle and stream for cleanup using a cleaner
        cleanableHandle = ResourceDealocator.register(this, hand -> JCublas2.cublasDestroy(hand), handle);
        cleanableStream = ResourceDealocator.register(this, str -> JCuda.cudaStreamDestroy(str), stream);
    }

    /**
     * Returns the CUBLAS handle managed by this class.
     * 
     * @return The {@link cublasHandle} for performing CUBLAS operations.
     */
    public cublasHandle get() {
        return handle;
    }

    /**
     * The stream associated with this handle.
     * 
     * @return The stream associated with this handle.
     */
    public cudaStream_t getStream() {
        return stream;
    }
    
    /**
     * Synchronizes the stream managed by this group. The {@code cudaStreamSynchronize}
     * method is called on each stream to ensure that all pending operations on 
     * not the stream are completed.  This method should be called before any methods
     * not using the handle, like Array::set or Array::get are called.
     * 
     */
    public void synch(){
        JCuda.cudaStreamSynchronize(stream);
    }
    /**
     * Closes the handle and stream, ensuring they are cleaned up.
     * This method is automatically called when the object is used in a try-with-resources statement.
     * It destroys the CUBLAS handle and the associated CUDA stream.
     */
    @Override
    public void close() {
        synch();
        cleanableHandle.clean();  // Clean up the CUBLAS handle
        cleanableStream.clean();  // Clean up the CUDA stream
    }
}
