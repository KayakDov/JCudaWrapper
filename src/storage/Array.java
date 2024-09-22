package storage;

import processSupport.ResourceDealocator;
import java.lang.ref.Cleaner;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import processSupport.Handle;

/**
 * Abstract class representing a GPU array that stores data on the GPU.
 * This class provides various utilities for managing GPU memory and copying data
 * between the host (CPU) and device (GPU).
 * 
 * <p>It supports different primitive types including byte, char, double, float, int,
 * long, and short. The derived classes should implement specific functionalities for each
 * type of data.</p>
 * 
 * <p>Note: This class is designed to be extended by specific array implementations.</p>
 * 
 * http://www.jcuda.org/tutorial/TutorialIndex.html#CreatingKernels
 * 
 * TODO:Implement methods with JBLAS as an aulternative for when there's no gpu.
 * 
 * @author E. Dov Neimand
 */
abstract class Array implements AutoCloseable {

    private final Cleaner.Cleanable cleanable;
    /**
     * The pointer to the array in gpu memory.
     */
    protected final CUdeviceptr pointer;
    /**
     * The length of the array.
     */
    public final int length;
    private final PrimitiveType type;

    /**
     * Enum representing different primitive types and their sizes.
     */
    public enum PrimitiveType {
        BYTE(Sizeof.BYTE), CHAR(Sizeof.CHAR), DOUBLE(Sizeof.DOUBLE), FLOAT(Sizeof.FLOAT), 
        INT(Sizeof.INT), LONG(Sizeof.LONG), SHORT(Sizeof.SHORT);

        public final int size;

        /**
         * Constructs a PrimitiveType enum.
         * 
         * @param numBytes The number of bytes per element.
         */
        private PrimitiveType(int numBytes) {
            this.size = numBytes;
        }
    }

    /**
     * Constructs an Array with the given GPU pointer, length, and element type.
     * 
     * @param p The pointer to the GPU memory allocated for this array.
     * @param length The length of the array.
     * @param type The type of elements in the array.
     * 
     * @throws IllegalArgumentException if the pointer is null or length is negative.
     */
    protected Array(CUdeviceptr p, int length, PrimitiveType type) {
        checkNull(p, type);        
        checkPos(length);
        
        this.pointer = p;
        this.length = length;
        this.type = type;

        // Register cleanup of GPU memory
        cleanable = ResourceDealocator.register(this, pointer -> JCuda.cudaFree(pointer), pointer);
    }

    /**
     * Creates a copy of this array.
     * @param handle The handle.
     * @return A new Array instance with the same data.
     */
    public abstract Array copy(Handle handle);

    /**
     * Returns a pointer to the element at the specified index in this array.
     * 
     * @param index The index of the element.
     * @return A CUdeviceptr pointing to the specified index.
     * 
     * @throws ArrayIndexOutOfBoundsException if the index is out of bounds.
     */
    protected CUdeviceptr pointer(int index) {
        checkPos(index);checkAgainstLength(index);
        
        return pointer.withByteOffset(index * type.size);
    }

    /**
     * Allocates GPU memory for an array of the specified size and type.
     * 
     * @param numElements The number of elements to allocate space for.
     * @param type The type of the array elements.
     * @return A CUdeviceptr pointing to the allocated memory.
     * 
     * @throws IllegalArgumentException if numElements is negative.
     */
    protected static CUdeviceptr empty(int numElements, PrimitiveType type) {
        checkPos(numElements);
               
        CUdeviceptr p = new CUdeviceptr();
        JCuda.cudaMalloc(p, numElements * type.size);
        return p;
    }

    /**
     * Copies data from a CPU array to a GPU array.
     * 
     * @param to The destination GPU array.
     * @param fromCPUArray The source CPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in the source array to start copying from.
     * @param length The number of elements to copy.
     * @param type The type of the elements.
     * @param handle The handle.
     * 
     * @throws IllegalArgumentException if any index is out of bounds or length is negative.
     */
    protected static void copy(Handle handle, Array to, Pointer fromCPUArray, int toIndex, int fromIndex, int length, PrimitiveType type) {
        checkPos(toIndex, fromIndex, length);
        to.checkAgainstLength(toIndex + length - 1); 
        
        JCuda.cudaMemcpyAsync(to.pointer.withByteOffset(toIndex * type.size),
                fromCPUArray.withByteOffset(fromIndex * type.size),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                handle.getStream()
        );
    }

    /**
     * Copies data from this GPU array to another GPU array.
     * 
     * @param to The destination GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * 
     * @throws IllegalArgumentException if any index is out of bounds or length is negative.
     */
    public void get(Handle handle, Array to, int toIndex, int fromIndex, int length) {
        checkPos(toIndex, fromIndex, length);
        to.checkAgainstLength(toIndex + length - 1);
        checkAgainstLength(fromIndex + length - 1);
        checkNull(to);
        
        JCuda.cudaMemcpyAsync(to.pointer(toIndex),
                pointer(fromIndex),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                handle.getStream()
        );
    }
    
    
    /**
     * Copies data from this GPU array to another GPU array.  Allows for multiple 
     * copying in parallel.
     * 
     * @param to The destination GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * 
     * @throws IllegalArgumentException if any index is out of bounds or length is negative.
     */
    public void get(Array to, int toIndex, int fromIndex, int length, Handle handle){
        JCuda.cudaMemcpyAsync(
                to.pointer(toIndex), 
                pointer(fromIndex),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToDevice, 
                handle.getStream());
    }
    
    
    /**
     * Copies data to this GPU array from another GPU array. Allows for multiple 
     * copying in parallel.
     * 
     * @param from The source GPU array.
     * @param toIndex The index in the destination array to start copying to.
     * @param fromIndex The index in this array to start copying from.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length is negative.
     */
    public void set(Handle handle, Array from, int toIndex, int fromIndex, int length){
        from.get(this, toIndex, fromIndex, length, handle);
    }

    /**
     * Copies data from this GPU array to a CPU array.
     * 
     * @param toCPUArray The destination CPU array.
     * @param toStart The starting index in the CPU array.
     * @param fromStart The starting index in this GPU array.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * 
     * @throws IllegalArgumentException if any index is out of bounds or length is negative.
     */
    protected void get(Handle handle, Pointer toCPUArray, int toStart, int fromStart, int length) {
        checkPos(toStart, fromStart, length);
        checkAgainstLength(fromStart + length - 1);
        //TODO:  cudaHostAlloc can be faster, but has risks.        
        JCuda.cudaMemcpyAsync(toCPUArray.withByteOffset(toStart * type.size),
                pointer(fromStart),
                length * type.size,
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                handle.getStream()
        );
    }

    /**
     * Copies data from a CPU array to this GPU array.
     * 
     * @param fromCPUArray The source CPU array.
     * @param toIndex The starting index in this GPU array.
     * @param fromIndex The starting index in the CPU array.
     * @param size The number of elements to copy.
     * @param handle The handle.
     * @throws ArrayIndexOutOfBoundsException if any index is out of bounds or size is negative.
     */
    protected void set(Handle handle, Pointer fromCPUArray, int toIndex, int fromIndex, int size) {
        checkPos(toIndex, fromIndex, size);
        checkAgainstLength(toIndex + size);
        
        copy(handle, this, fromCPUArray, toIndex, fromIndex, size, type);
    }

    
    /**
     * Copies data from a CPU array to this GPU array starting from the beginning.
     * 
     * @param fromCPUArray The source CPU array.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws ArrayIndexOutOfBoundsException if length is negative.
     */
    protected void set(Handle handle, Pointer fromCPUArray, int length) {
        
        set(handle, fromCPUArray, 0, 0, length);
    }

    /**
     * Copies data from a CPU array to this GPU array with a specified starting index.
     * 
     * @param fromCPUArray The source CPU array.
     * @param toIndex The starting index in this GPU array.
     * @param length The number of elements to copy.
     * @param handle The handle.
     * @throws IllegalArgumentException if any index is out of bounds or length is negative.
     */
    protected void set(Handle handle, Pointer fromCPUArray, int toIndex, int length) {
        checkPos(toIndex, length);
        checkAgainstLength(toIndex + length);
        
        set(handle, fromCPUArray, toIndex, 0, length);
    }
    

    /**
     * Frees the GPU memory allocated for this array.
     * This method is invoked automatically when the object is closed.
     */
    @Override
    public void close() {
        cleanable.clean();
    }
    
    
    /**
     * Sets the contents of this array to 0.
     * @param  handle The handle.
     * @return this.
     */
    public Array fill0(Handle handle){
        JCuda.cudaMemsetAsync(pointer, 0, length*type.size, handle.getStream());
        return this;
    }

    
    
    /**
     * A pointer to a singleton array containing d.
     * @param d A double that needs a pointer.
     * @return A pointer to a singleton array containing d.
     */
   protected static Pointer cpuPointer(double d){
       return Pointer.to(new double[]{d});
   }
   
   
    /**
     * Checks that all the numbers are positive.  An exception is thrown if not.
     * @param maybePos A number that might be positive.
     */
    protected static void checkPos(int... maybePos){
        checkLowerBound(0, maybePos);
    }
    
    /**
     * Checks that all the numbers are greater than or equal to the lower bound.  
     * An exception is thrown if not.
     * @param bound The lower bound.
     * @param needsCheck A number that might be greater than the lower bound.
     */
    protected static void checkLowerBound(int bound, int... needsCheck){
        if(Arrays.stream(needsCheck).anyMatch(l -> bound > l))
            throw new ArrayIndexOutOfBoundsException();
    }
    
    /**
     * Checks if any of the indeces are out of bounds, that is, greater than or 
     * equal to the length.
     * @param maybeInBounds A number that might or might not be out of bounds. 
     */
    protected void checkAgainstLength(int ... maybeInBounds){
        checkUpperBound(length, maybeInBounds);
    }
    
    /**
     * Checks if any of the indeces are out of bounds, that is, greater than or 
     * equal to the upper bound
     * @param bound ALl elements passed must be less than this element..
     * @param maybeInBounds A number that might or might not be out of bounds. 
     */
    protected void checkUpperBound(int bound, int ... maybeInBounds){
        for(int needsChecking: maybeInBounds)
            if(needsChecking >= bound)
                throw new ArrayIndexOutOfBoundsException(needsChecking + " is greater than the bound " + bound);
    }
       
    
    /**
     * Checks if any of the objects are null.  If they are a 
     * NullPointerException is thrown.
     * @param maybeNull Objects that might be null.
     */
    protected static void checkNull(Object... maybeNull){
        if(Arrays.stream(maybeNull).anyMatch(l -> l == null))
            throw new NullPointerException();
    }
    
    
}





//    /**
//     * Uses a kernel to find the argmax and the maximum element for the array.
//     * @param inc The stride length of the array.
//     * @return The maximum element and its index in the array.
//     */
//    public double[] findMax(Handle handle, int inc) {
//
//        DArray result = DArray.empty(2);
//        
//        runKernel1d("src/kernels/find_max.ptx",
//                "find_max",
//                Sizeof.DOUBLE +Sizeof.INT,
//                new Pointer[]{
//                    Pointer.to(result.pointer),
//                    Pointer.to(new int[]{inc})
//                }
//        );
//        return result.get(handle);
//    }
//    
//    public static void main(String[] args) {
//        Handle hand = new Handle();
//        DArray da = new DArray(hand, 5, 3, 10, 9);
//        double[] result = da.findMax(hand, 1);
//        
//        
//        System.out.println(Arrays.toString(result));
//        
//        hand.close();
//        da.close();
//    }
//
//    
//
//    /**
//     * TODO: Check the run kernel methods.  They may not work.
//     * 
//     * Runs a CUDA kernel by loading the specified PTX file and executing the
//     * function with the provided arguments on the GPU. The first argumrnt of
//     * the kernel should be this pointer, and the 2nd argument should be this
//     * length.
//     *
//     * @param kernelFileName The path to the PTX file containing the compiled
//     * CUDA kernel. This file must be compiled beforehand using the `nvcc`
//     * compiler.
//     * @param kernelFunctionName The name of the kernel function to execute.
//     * This should match the function name defined in the .PTX file.
//     * @param args The arguments to pass to the CUDA kernel. These arguments
//     * will be passed in the order they appear. The kernel's expected parameters
//     * must match the types and order of these arguments.
//     *
//     * Example usage:      <pre>
//     * {@code
//     * runKernel("src/gpu/MatrixFillKernel.ptx", "fillMatrix", Pointer.to(new double[]{scalar}));
//     * }
//     * </pre>
//     *
//     * Process: 1. The PTX file is loaded, and a module is created from it. 2. A
//     * handle to the kernel function (`CUfunction`) is obtained. 3. Grid and
//     * block sizes are calculated based on the number of elements to process. 4.
//     * The provided arguments are packed into a `Pointer` array and combined
//     * with additional internal arguments such as the device memory pointer and
//     * the size of the matrix. 5. The kernel is launched using `cuLaunchKernel`,
//     * which runs the function on the GPU with the specified grid and block
//     * sizes. 6. Finally, the method synchronizes the context to ensure that the
//     * kernel execution completes before continuing.
//     */
//    private void runKernel(String kernelFileName, String kernelFunctionName, Pointer[] args,
//            int gridSizeX, int gridSizeY, int blockSizeX, int blockSizeY, int sharedMemSize) {
//        checkPos(gridSizeX, gridSizeY, blockSizeX, blockSizeY);
//        
//        // Load the PTX file
//        CUmodule module = new CUmodule();
//        JCudaDriver.cuModuleLoad(module, kernelFileName);
//
//        // Obtain a function handle for the kernel
//        CUfunction function = new CUfunction();
//        JCudaDriver.cuModuleGetFunction(function, module, kernelFunctionName);
//
//        // Prepare kernel parameters
//        Pointer kernelParameters = Pointer.to(args);
//
//        // Launch the kernel
//        JCudaDriver.cuLaunchKernel(function,
//                gridSizeX, gridSizeY, 1, // Grid size in 2D
//                blockSizeX, blockSizeY, 1, // Block size in 2D
//                sharedMemSize, null, // Shared memory size and stream
//                kernelParameters, null);
//
//        // Synchronize to wait for the kernel to finish
//        JCudaDriver.cuCtxSynchronize();
//    }
//
//    /**
//     * Runs the requested kernel on an array.
//     *
//     * @param kernelFileName The name and relative location of the kernel file.
//     * @param kernelFunctionName The name of the function in the kernel.
//     * @param args Sequentially correct arguments not including pointer and
//     * length.
//     */
//    private void runKernel1d(String kernelFileName, String kernelFunctionName, int extMemPerBlock, Pointer... args) {
//
//        int blockSizeX = 256;
//        int gridSizex = (length + blockSizeX - 1) / blockSizeX;  // Ensures all elements are covered
//        int blockSizeY = 1;
//        int gridSizeY = 1;
//        
//        int sharedMemSize = blockSizeX * extMemPerBlock; 
//
//
//
//        Pointer[] argsWithLocal = new Pointer[args.length + 2];
//        System.arraycopy(args, 0, argsWithLocal, 2, args.length);
//        argsWithLocal[0] = Pointer.to(pointer);
//        argsWithLocal[1] = Pointer.to(new int[]{length});
//
//        runKernel(kernelFileName, kernelFunctionName, argsWithLocal, gridSizex, gridSizeY, blockSizeX, blockSizeY, sharedMemSize);
//    }
//
//    /**
//     * Runs a kernel on 2d data.
//     *
//     * @param kernelFileName The name of the file the kernel is in.
//     * @param kernelFunctionName The name of the kernel function.
//     * @param width The width of the matrix.
//     * @param height The height of the matrix.
//     * @param args The sequentially correct arguments after pointer, height,
//     * width.
//     */
//    private void runKernel2d(String kernelFileName, String kernelFunctionName, int height, int width, Pointer... args) {
//
//        Pointer[] argsWithLocal = new Pointer[args.length + 3];
//        System.arraycopy(args, 0, argsWithLocal, 3, args.length);
//        argsWithLocal[0] = Pointer.to(pointer);
//        argsWithLocal[1] = Pointer.to(new int[]{height});
//        argsWithLocal[2] = Pointer.to(new int[]{width});
//
//        int blockSize = 16; // Threads per block in X direction (width)        
//
//        int gridSizeX = (width + blockSize - 1) / blockSize;
//        int gridSizeY = (height + blockSize - 1) / blockSize;
//
//        runKernel(kernelFileName, kernelFunctionName, argsWithLocal, gridSizeX, gridSizeY, blockSize, blockSize);
//
//    }
