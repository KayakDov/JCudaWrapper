package main;


import algebra.Matrix;
import algebra.Vector;
import processSupport.Handle;
import storage.DArray;

public class VectorTest {

    public static void main(String[] args) {
        // Initialize a JCublas handle (mock or actual handle based on your environment)
        Handle handle = new Handle(); 
        
        
        Matrix m = new Matrix(handle, new double[][]{{1,2},{3,4}});
        
        m.setRow(1, new double[]{5,6});
        
        System.out.println(m.toString());
        
        

        // Test case 1: Create a vector from an array and print values
//        double[] array = {1.0, 2.0, 3.0, 4.0, 5.0};
//        Vector vector1 = new Vector(array, handle);
//        System.out.println("Test Case 1: Vector from array:");
//        printVector(vector1);
//
//        // Test case 2: Create an empty vector and fill it with a value
//        Vector vector2 = new Vector(5, handle);
//        vector2.fill(10.0);
//        System.out.println("\nTest Case 2: Empty vector filled with 10.0:");
//        printVector(vector2);
//
//        // Test case 3: Perform vector addition
//        Vector sumVector = vector1.add(vector2);
//        System.out.println("\nTest Case 3: Sum of vector1 and vector2:");
//        printVector(sumVector);
//
//        // Test case 4: Multiply vector by scalar
//        Vector scaledVector = vector1.mapMultiply(2.0);
//        System.out.println("\nTest Case 4: Vector1 multiplied by scalar 2.0:");
//        printVector(scaledVector);
//
//        // Test case 5: Dot product of two vectors
//        double dotProduct = vector1.dotProduct(vector2);
//        System.out.println("\nTest Case 5: Dot product of vector1 and vector2:");
//        System.out.println(dotProduct);
//
//        // Test case 6: Element-wise multiplication of two vectors
//        Vector ebeMultVector = vector1.ebeMultiply(vector2);
//        System.out.println("\nTest Case 6: Element-wise multiplication of vector1 and vector2:");
//        printVector(ebeMultVector);
//
//        // Test case 7: Subtract two vectors
//        Vector subtractedVector = vector2.subtract(vector1);
//        System.out.println("\nTest Case 7: Subtract vector1 from vector2:");
//        printVector(subtractedVector);
//        
//        // Test case 8: Norm of the vector
//        double norm = vector1.getNorm();
//        System.out.println("\nTest Case 8: Norm of vector1:");
//        System.out.println(norm);
//
//        // Close vectors (AutoCloseable)
//        vector1.close();
//        vector2.close();
//        sumVector.close();
//        scaledVector.close();
//        ebeMultVector.close();
//        subtractedVector.close();
    }

    // Helper method to print the contents of a vector
    private static void printVector(Vector vector) {
        int dimension = vector.getDimension();
        for (int i = 0; i < dimension; i++) {
            System.out.print(vector.getEntry(i) + " ");
        }
        System.out.println();
    }
}
