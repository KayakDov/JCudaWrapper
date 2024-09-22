package processSupport;

import java.lang.ref.Cleaner;
import java.util.Arrays;
import java.util.stream.IntStream;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import processSupport.ResourceDealocator;
import storage.DArray;
import storage.DSingleton;

/**
 * A utility class for managing a group of CUDA streams with an immutable list
 * of streams. This class allows the creation, iteration, and synchronized
 * closure of a set of CUDA streams. It also implements {@link AutoCloseable}
 * for automatic resource cleanup, ensuring that the streams are destroyed when
 * they are no longer needed.
 *
 
 * Usage:
 * <pre>
 * try (StreamGroup streamGroup = new StreamGroup(10)) {
 *     streamGroup.forEach(stream -> {
 *         // Perform operations with the stream
 *     });
 * }
 * </pre>
 
 *
 * <p>
 * Note: The streams are automatically synchronized and destroyed when the
 * {@link #close()} method is called, either explicitly or implicitly via the
 * try-with-resources statement.</p>
 *
 * @author edov
 */
public class StreamGroup implements AutoCloseable {

    /**
     * The {@link Cleaner.Cleanable} instance for automatic resource cleanup.
     */
    private final Cleaner.Cleanable cleanableStreams;
    private Cleaner.Cleanable cleanableHandles;

    /**
     * The array of CUDA streams managed by this class.
     */
    private final cudaStream_t[] streams;
    private cublasHandle[] handles;

    /**
     * An interface to define operations that return a value from streams.
     */
    public interface DHandGetter {
        /**
         * Gets a value of type {@code T} based on the stream and its associated handle.
         * @param i The index of the stream.
         * @param handle The handle associated with the stream.
         * @param result Where to store the result.
         */
        void apply(int i,cublasHandle handle, DSingleton result);
    }
    
    /**
     * An interface to define operations that return a value from streams.
     */
    public interface DStreamGetter {
        /**
         * Gets a value of type {@code T} based on the stream and its associated handle.
         * @param i The index of the stream.
         * @param stream The CUDA stream.         
         * @param result Where to store the result.
         */
        void apply(int i, cudaStream_t stream, DSingleton result);
    }

    /**
     * An interface to define operations to consume streams and handles.
     */
    public interface StreamConsumer {
        /**
         * Performs an operation using the stream and its associated handle.
         * @param i The index of the stream.
         * @param stream The CUDA stream.         
         */
        void accept(int i, cudaStream_t stream);
    }

    
    /**
     * An interface to define operations to consume streams and handles.
     */
    public interface HandConsumer {
        /**
         * Performs an operation using the stream and its associated handle.
         * @param i The index of the stream.
         * @param handle The handle.         
         */
        void accept(int i,cublasHandle handle);
    }

    
    /**
     * Constructs a new {@code StreamGroup} with a specified number of CUDA streams.
     *
     * @param size The number of CUDA streams to create.
     */
    public StreamGroup(int size) {
        streams = new cudaStream_t[size];
        Arrays.setAll(streams, i -> new cudaStream_t());
        for (cudaStream_t str : streams) 
            JCuda.cudaStreamCreate(str);
        
        cleanableStreams = ResourceDealocator.register(this, stream -> JCuda.cudaStreamDestroy(stream), streams);
    }

    /**
     * Associates a handle with each stream for methods that need handles.
     * This method initializes handles only once.
     */
    public void setHandles() {
        if (handles == null) {
            handles = new cublasHandle[streams.length];
            Arrays.setAll(handles, i -> new cublasHandle());
            for (int i = 0; i < handles.length; i++) {
                JCublas2.cublasCreate(handles[i]);
                JCublas2.cublasSetStream(handles[i], streams[i]);
            }
            cleanableHandles = ResourceDealocator.register(this, handle -> JCublas2.cublasDestroy(handle), handles);
        }
    }

    /**
     * Performs the specified operation on each CUDA stream in this group with associated handles.
     *
     * @param f The operation to be performed for each stream and its handle.
     */
    public void runParallelStreams(StreamConsumer f) {
        IntStream.range(0, streams.length).parallel().forEach(i -> f.accept(i, streams[i]));
        synch();
    }
    
    /**
     * Performs the specified operation on each CUDA stream in this group with associated handles.
     *
     * @param f The operation to be performed for each stream and its handle.
     */
    public void runParallelHandles(HandConsumer f) {
        IntStream.range(0, streams.length).parallel().forEach(i -> f.accept(i, handles[i]));
        synch();
    }


    /**
     * Performs the specified operation on each CUDA stream in this group with associated handles
     * and returns the results.
     *
     * @param f The operation to be performed for each stream and its handle.
     * @return A list of results from each operation.
     */
    public DArray getParallelStreams(DStreamGetter f) {
        
        DArray results = DArray.empty(streams.length);
        
        IntStream.range(0, streams.length).parallel().forEach(i -> f.apply(i, streams[i], results.get(i)));
        
        synch();
        
        return results;
    }
    
    
    /**
     * Performs the specified operation on each CUDA stream in this group with associated handles
     * and returns the results.
     *
     * @param f The operation to be performed for each stream and its handle.
     * @return A list of results from each operation.
     */
    public DArray getParallelHandles(DHandGetter f) {
        
        DArray results = DArray.empty(streams.length);
        
        IntStream.range(0, streams.length).parallel().forEach(i -> f.apply(i, handles[i], results.get(i)));
        
        synch();
        
        return results;
    }

    /**
     * Synchronizes all CUDA streams managed by this group. The {@code cudaStreamSynchronize}
     * method is called on each stream to ensure that all pending operations on the stream are completed.
     */
    public void synch(){
        for (cudaStream_t stream : streams) 
            JCuda.cudaStreamSynchronize(stream);
    }
    
    /**
     * Destroys all CUDA streams managed by this group. The {@code cudaStreamSynchronize}
     * method is called on each stream to ensure that all pending operations on the stream are completed
     * before the stream is destroyed.
     *
     * <p>This method is automatically called when the object is used in a try-with-resources statement.</p>
     */
    @Override
    public void close() {
        synch();
        cleanableStreams.clean();
        if (cleanableHandles != null) {
            cleanableHandles.clean();
        }
    }
}
