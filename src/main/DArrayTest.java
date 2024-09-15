package main;

import storage.DArray;
import processSupport.Handle;
import java.util.Arrays;
import jcuda.runtime.JCuda;

public class DArrayTest {

    public static void main(String[] args) throws Exception {

        // Initialize the GPU handle for all operations
        Handle handle = new Handle();

        // Run tests
        boolean allTestsPassed = true;
        allTestsPassed &= testConstructorAndGet();
        allTestsPassed &= testSet();
        allTestsPassed &= testCopy();
        allTestsPassed &= testDotProduct(handle);
        allTestsPassed &= testAddToMe(handle);
        allTestsPassed &= testMultMe(handle);
        allTestsPassed &= testMatrixMultiplication(handle);
        allTestsPassed &= testSubArray();
        allTestsPassed &= testSetAndGetByIndex();

        // Cleanup GPU handle
        handle.close();

        if (allTestsPassed) {
            System.out.println("All tests passed.");
        } else {
            System.out.println("Some tests failed.");
        }
    }

    private static boolean arraysEqual(double[] actual, double[] expected, double tolerance) {
        if (actual.length != expected.length) return false;
        
        for (int i = 0; i < actual.length; i++) 
            if (Math.abs(actual[i] - expected[i]) > tolerance) return false;
        
        return true;
    }

    
    private static boolean testConstructorAndGet() {
        System.out.println("Testing DArray constructor and get()...");
        try {
        
            double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
            
            DArray dArray = new DArray(values);
            
            double[] result = dArray.get();
            
            boolean passed = arraysEqual(result, values, 1e-9);
            
            System.out.println("Test passed: " + passed);
            
            dArray.close();
            return passed;
        
        } catch (Exception e) {
            System.err.println("Error in testConstructorAndGet: " + e.getMessage());
            return false;
        }
    }

    private static boolean testSet() {
        System.out.println("Testing DArray set() method...");
        try {
            double[] initialValues = {1.0, 2.0, 3.0, 4.0, 5.0};
            double[] result;
            
            try (DArray dArray = new DArray(initialValues)) {
                double[] newValues = {6.0, 7.0, 8.0};
                dArray.set(newValues, 2);
                result = dArray.get();
            }

            double[] expected = {1.0, 2.0, 6.0, 7.0, 8.0};
            boolean passed = arraysEqual(result, expected, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testSet: " + e.getMessage());
            return false;
        }
    }

    private static boolean testCopy() {
        System.out.println("Testing DArray copy()...");
        try {
            double[] values = {1.0, 2.0, 3.0, 4.0, 5.0};
            DArray dArray = new DArray(values);

            DArray dArrayCopy = dArray.copy();
            double[] result = dArrayCopy.get();
            dArray.close();
            dArrayCopy.close();

            boolean passed = arraysEqual(result, values, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testCopy: " + e.getMessage());
            throw e;
            
        }
    }

    private static boolean testDotProduct(Handle handle) {
        System.out.println("Testing DArray dot() method...");
        try {
            DArray dArray1 = new DArray(1.0, 2.0, 3.0, 4.0, 5.0);
            DArray dArray2 = new DArray(5.0, 4.0, 3.0, 2.0, 1.0);

            double dotProduct = dArray1.dot(handle, dArray2, 1, 1);
            dArray1.close();
            dArray2.close();

            double expected = 35;
            boolean passed = Math.abs(dotProduct - expected) < 1e-9;
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testDotProduct: " + e.getMessage());
            return false;
        }
    }

    private static boolean testAddToMe(Handle handle) {
        System.out.println("Testing DArray addToMe() method...");
        try {
            DArray dArray1 = new DArray(1.0, 2.0, 3.0, 4.0, 5.0);
            DArray dArray2 = new DArray(5.0, 4.0, 3.0, 2.0, 1.0);

            dArray1.addToMe(handle, 1.0, dArray2, 1, 1);
            double[] result = dArray1.get();
            dArray1.close();
            dArray2.close();

            double[] expected = {6.0, 6.0, 6.0, 6.0, 6.0};
            boolean passed = arraysEqual(result, expected, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testAddToMe: " + e.getMessage());
            return false;
        }
    }

    private static boolean testMultMe(Handle handle) {
        System.out.println("Testing DArray multMe() method...");
        try {
            DArray dArray = new DArray(6.0, 6.0, 9.0, 9.0, 9.0);

            dArray.multMe(handle, 2, 2);
            double[] result = dArray.get();
            dArray.close();

            double[] expected = {12.0, 6.0, 18.0, 9.0, 18.0};
            boolean passed = arraysEqual(result, expected, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testMultMe: " + e.getMessage());
            return false;
        }
    }

    private static boolean testMatrixMultiplication(Handle handle) {
        System.out.println("Testing DArray multMatMat() method...");
        try {
            DArray a = new DArray(1, 2,   3, 4,   5, 6);  // A is 2x3
            DArray b = new DArray(7, 8, 9,   10, 11, 12); // B is 3x2
            DArray c = DArray.empty(4);           // C is 2x2

            c.multMatMat(handle, false, false, 2, 2, 3, 1, a, 2, b, 3, 0, 2);
            double[] result = c.get();
            a.close();
            b.close();
            c.close();

            double[] expected = {76, 100, 103, 136};  // Expected result of A * B
            boolean passed = arraysEqual(result, expected, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testMatrixMultiplication: " + e.getMessage());
            return false;
        }
    }

    private static boolean testSubArray() {
        System.out.println("Testing DArray subArray() method...");
        try {
            DArray dArray = new DArray(1.0, 2.0, 3.0, 4.0, 5.0);
            DArray subArray = dArray.subArray(1, 3);
            double[] result = subArray.get();
            dArray.close();
            subArray.close();

            double[] expected = {2.0, 3.0, 4.0};
            boolean passed = arraysEqual(result, expected, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testSubArray: " + e.getMessage());
            return false;
        }
    }

    private static boolean testSetAndGetByIndex() {
        System.out.println("Testing DArray set(int index, double val) and get(int index)...");
        try {
            DArray dArray = new DArray(1.0, 2.0, 3.0, 4.0, 5.0);
            dArray.set(0, 10.0);
            double val = dArray.get(0, 1)[0];
            dArray.close();

            boolean passed = Math.abs(val - 10.0) < 1e-9;
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testSetAndGetByIndex: " + e.getMessage());
            return false;
        }
    }
}
