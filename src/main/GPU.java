package main;

import jcuda.CudaException;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.runtime.JCuda;

/**
 * Computer wide GPU properties and set up.
 * @author E. Dov Neimand
 */
public class GPU {
    private static boolean useGPU;
    
    

    static {
        useGPU = checkGPUAvailability();
        if (useGPU) {
            JCublas.cublasInit();
            JCuda.setExceptionsEnabled(true);
            JCublas.setExceptionsEnabled(true);
            JCudaDriver.setExceptionsEnabled(true);

        }
    }
    
    
    /**
     * Checks if the GPU is accessible and works. If not, use the CPU.
     *
     * @return true if GPU is available, false otherwise.
     */
    private static boolean checkGPUAvailability() {
        try {
            int[] deviceCount = new int[1];
            JCuda.cudaGetDeviceCount(deviceCount);
            return deviceCount[0] > 0;
        } catch (CudaException e) {
            System.out.println("GPU is not available. Falling back to CPU.");
            return false;
        }
    }
    
    
    /**
     * Should the GPU or CPU be used for matrix multiplication.
     *
     * @param useGPU True to use the gpu and flase otherwise. Setting this to
     * false will disable the construction of GPUMatrix.
     */
    public static void setUseGPU(boolean useGPU) {
        GPU.useGPU = useGPU;
        if (useGPU && !checkGPUAvailability()) {
            GPU.useGPU = false;
            System.out.println("GPU not available.");
        }
    }
    
    

    /**
     * Checks if the useGPU flag is true or false.
     *
     * @return The useGPU flag.
     */
    public static boolean IsAvailable() {
        return useGPU;
    }
}
