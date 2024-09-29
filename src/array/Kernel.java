package array;

import java.io.File;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

/**
 * The {@code Kernel} class is a utility for managing and executing CUDA kernels 
 * using JCuda. It handles loading CUDA modules, setting up functions, and executing
 * them with specified parameters.
 * <p>
 * This class is designed to simplify the process of launching CUDA kernels 
 * with a given input and output {@code DArray}.
 * </p>
 * 
 * <p>
 * Example usage:
 * <pre>
 * {@code
 * Kernel kernel = new Kernel("atan2.ptx", "atan2xy", n);
 * DArray result = kernel.run(inputArray, outputArray, numOperations);
 * }
 * </pre>
 * </p>
 * 
 * @author E. Dov Neimand
 */
public class Kernel {

    /**
     * The CUDA function handle for the loaded kernel.
     */
    private final CUfunction function;

    /**
     * The number of blocks to be launched in the grid.
     */
    private final int gridSize;

    /**
     * The number of threads per block to be used in kernel execution.
     */
    private final static int BLOCK_SIZE = 256;

    /**
     * Constructs a {@code Kernel} object that loads a CUDA module from a given file
     * and retrieves a function handle for the specified kernel function.
     *
     * @param fileName The name of the PTX file containing the CUDA kernel (e.g., "atan2.ptx").
     * @param functionName The name of the kernel function to execute (e.g., "atan2xy").
     * @param n The total number of operations to determine the grid size.
     */
    public Kernel(String fileName, String functionName, int n) { 
        CUmodule module = new CUmodule();
        JCudaDriver.cuModuleLoad(module, "src" + File.separator + "kernels" + File.separator + fileName);
        
        function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, functionName);

        gridSize = (int) Math.ceil((double) n / BLOCK_SIZE);
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on a specified stream.
     * Note, a stream is generated for this method, so be sure that the data
     * is synchronized before and after.
     * @param input The {@code DArray} representing the input data to be processed by the kernel.
     * @param output The {@code DArray} representing the output data where results will be stored.     
     * @return The {@code DArray} containing the processed results.
     */
    public DArray map(DArray input, DArray output) {
        System.out.println("storage.Kernel.map() " + input);
        System.out.println("storage.Kernel.map() " + output.length);
        
        Pointer kernelParameters = Pointer.to(
                IArray.cpuPointer(output.length), // Pointer to the number of elements
                Pointer.to(input.pointer),        // Pointer to input data in device memory
                Pointer.to(output.pointer)        // Pointer to output data in device memory
        );

        // Launch the CUDA kernel with the specified stream
        JCudaDriver.cuLaunchKernel(
                function,
                gridSize, 1, 1, // Grid size (number of blocks)
                BLOCK_SIZE, 1, 1, // Block size (number of threads per block)
                0, null, // Shared memory size and the specified stream
                kernelParameters, null // Kernel parameters
        );

        return output;
    }
    
    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on a specified stream.
     * @param input The {@code DArray} representing the input data to be processed by the kernel.
     * @param numOperation The number of operations or elements to process.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapping(DArray input, int numOperation) {
        return map(input, DArray.empty(numOperation));
    }

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified stream.
     * @param input The {@code DArray} representing the input data to be processed by the kernel.
     * @param stream The CUDA stream in which to execute the kernel (can be {@code null} for default stream).
     * @return The {@code DArray} containing the processed results.
     */    
    public DArray mapToSelf(DArray input) {
        return map(input, input);
    }
}
