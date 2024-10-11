package array;

import algebra.Vector;
import array.IArray;
import java.io.File;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

/**
 * The {@code Kernel} class is a utility for managing and executing CUDA kernels
 * using JCuda. It handles loading CUDA modules, setting up functions, and
 * executing them with specified parameters.
 * 
 * Helper methods include https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
 * <p>
 * This class is designed to simplify the process of launching CUDA kernels with
 * a given input and output {@code DArray}.
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

    private int batchSize;

    /**
     * Constructs a {@code Kernel} object that loads a CUDA module from a given
     * file and retrieves a function handle for the specified kernel function.
     *
     * @param fileName The name of the PTX file containing the CUDA kernel
     * (e.g., "atan2.ptx"). If the file is in the kernel folder, then no path is
     * required.
     * @param functionName The name of the kernel function to execute (e.g.,
     * "atan2xy").
     * @param n The total number of operations to determine the grid size.
     */
    public Kernel(String fileName, String functionName, int n) {
        CUmodule module = new CUmodule();

        File ptxFile = new File("src" + File.separator + "kernels" + File.separator+ "ptx" + File.separator + fileName);
        if (!ptxFile.exists())
            throw new RuntimeException("Kernel file not found: " + ptxFile.getAbsolutePath());

        JCudaDriver.cuModuleLoad(module, ptxFile.getAbsolutePath());

        function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, functionName);

        if (function == null) {
            throw new RuntimeException("Failed to load kernel function");
        }

        gridSize = (int) Math.ceil((double) n / BLOCK_SIZE);

        batchSize = n;
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param incInput The increment for the input array.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @param incOutput The increment for the output array.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray map(DArray input, int incInput, DArray output, int incOutput) {

        Pointer kernelParameters = Pointer.to(
                Pointer.to(input.pointer),
                IArray.cpuPointer(incInput),
                Pointer.to(output.pointer),
                IArray.cpuPointer(incOutput),
                IArray.cpuPointer(batchSize)
        );

        int result = JCudaDriver.cuLaunchKernel(
                function,
                gridSize, 1, 1, // Grid size (number of blocks)
                BLOCK_SIZE, 1, 1, // Block size (number of threads per block)
                0, null, // Shared memory size and the specified stream
                kernelParameters, null // Kernel parameters
        );
        checkResult(result);

        JCudaDriver.cuCtxSynchronize();

        return output;
    }

    /**
     * Checks for error messages, and throws an exception if the operation
     * failed.
     *
     * @param result The result of a cuLaunch.
     */
    private void checkResult(int result) {
        if (result != CUresult.CUDA_SUCCESS) {
            String[] errorMsg = new String[1];
            JCudaDriver.cuGetErrorString(result, errorMsg);
            throw new RuntimeException("CUDA error during : " + errorMsg[0]);
        }
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray map(DArray input, DArray output) {
        return map(input, 1, output, 1);
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray map(Vector input, Vector output) {
        return map(input.dArray(), input.inc, output.dArray(), output.inc);
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param numOperation The number of operations or elements to process.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapping(DArray input, int numOperation) {
        return map(input, DArray.empty(numOperation));
    }

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified
     * stream.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapToSelf(DArray input) {
        return map(input, input);
    }

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified
     * stream.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapToSelf(DArray input, int inc) {
        return map(input, inc, input, inc);
    }

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified
     * stream.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapToSelf(Vector input) {
        return map(input, input);
    }
}
