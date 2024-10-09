package main;

import algebra.Matrix;
import algebra.Vector;
import java.awt.Point;
import java.awt.geom.Point2D;
import java.util.Arrays;
import array.DArray;
import array.DArray2d;
import array.IArray;
import resourceManagement.EigenSupport;
import resourceManagement.Handle;
import resourceManagement.JacobiParams;

/**
 * Tests the DArray class.
 *
 * @author E. Dov Neimand
 */
public class DArrayTest {

    public static Handle handle;

    /**
     * Runs the tests.
     *
     * @param args Not used.
     */
    public static void main(String[] args) {

        // Initialize the GPU handle for all operations
        handle = new Handle();

        // Run tests
        boolean allTestsPassed = true;
        allTestsPassed = testConstructorAndGet()
                & testSet()
                & testCopy()
                & testDotProduct()
                & testAddToMe()
                & testMultMe()
                & testMatrixMultiplication()
                & testSubArray()
                & testSetAndGetByIndex()
                & testAtan2()
                & testMultMatMatBatched()
                & testEigen();

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

            DArray dArray = new DArray(handle, values);

            double[] result = dArray.get(handle);

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

            try (DArray dArray = new DArray(handle, initialValues)) {
                double[] newValues = {6.0, 7.0, 8.0};
                dArray.set(handle, newValues, 2);
                result = dArray.get(handle);
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
            DArray dArray = new DArray(handle, values);

            DArray dArrayCopy = dArray.copy(handle);
            double[] result = dArrayCopy.get(handle);
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

    private static boolean testDotProduct() {
        System.out.println("Testing DArray dot() method...");
        try {
            DArray dArray1 = new DArray(handle, 1.0, 2.0, 3.0, 4.0, 5.0);
            DArray dArray2 = new DArray(handle, 5.0, 4.0, 3.0, 2.0, 1.0);

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

    private static boolean testAddToMe() {
        System.out.println("Testing DArray addToMe() method...");
        try {
            DArray dArray1 = new DArray(handle, 1.0, 2.0, 3.0, 4.0, 5.0);
            DArray dArray2 = new DArray(handle, 5.0, 4.0, 3.0, 2.0, 1.0);

            dArray1.addToMe(handle, 1.0, dArray2, 1, 1);
            double[] result = dArray1.get(handle);
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

    private static boolean testMultMe() {
        System.out.println("Testing DArray multMe() method...");
        try {
            DArray dArray = new DArray(handle, 6.0, 6.0, 9.0, 9.0, 9.0);

            dArray.multMe(handle, 2, 2);
            double[] result = dArray.get(handle);
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

    private static boolean testMatrixMultiplication() {
        System.out.println("Testing DArray multMatMat() method...");
        try {
            DArray a = new DArray(handle, 1, 2, 3, 4, 5, 6);  // A is 2x3
            DArray b = new DArray(handle, 7, 8, 9, 10, 11, 12); // B is 3x2
            DArray c = DArray.empty(4);           // C is 2x2

            c.multMatMat(handle, false, false, 2, 2, 3, 1, a, 2, b, 3, 0, 2);
            double[] result = c.get(handle);
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
            DArray dArray = new DArray(handle, 1.0, 2.0, 3.0, 4.0, 5.0);
            DArray subArray = dArray.subArray(1, 3);
            double[] result = subArray.get(handle);
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
            DArray dArray = new DArray(handle, 1.0, 2.0, 3.0, 4.0, 5.0);
            dArray.set(handle, 0, 10.0);
            double val = dArray.get(handle, 0, 1)[0];
            dArray.close();

            boolean passed = Math.abs(val - 10.0) < 1e-9;
            System.out.println("Test passed: " + passed);
            return passed;
        } catch (Exception e) {
            System.err.println("Error in testSetAndGetByIndex: " + e.getMessage());
            return false;
        }
    }

    /**
     * Tests the DArray atan2() method. This method tests the a^tan2
     * functionality that computes angles from pairs of x, y coordinates.
     *
     * @param handle The GPU handle used for the test.
     * @return true if the test passes, false otherwise.
     */
    private static boolean testAtan2() {
        System.out.println("Testing DArray atan2() method...");
        try {

            Point2D[] points = new Point2D[]{new Point(1, 0), new Point(-1, 0), new Point(0, 1), new Point(0, -1)};

            double[] vectors = new double[2 * points.length];
            for (int i = 0; i < points.length; i++) {
                vectors[2 * i] = points[i].getX();
                vectors[2 * i + 1] = points[i].getY();
            }

            DArray input = new DArray(handle, vectors);

            double[] expectedAngles = new double[points.length];
            Arrays.setAll(expectedAngles, i -> Math.atan2(points[i].getY(), points[i].getX()));

            DArray result = DArray.empty(points.length).atan2(input);

            double[] angles = result.get(handle);

            System.out.println(Arrays.toString(angles));

            input.close();
            result.close();

            boolean passed = arraysEqual(angles, expectedAngles, 1e-9);
            System.out.println("Test passed: " + passed);
            return passed;

        } catch (Exception e) {
            System.err.println("Error in testAtan2: " + e.getMessage());
            return false;
        }
    }

    public static boolean testMultMatMatBatched() {

        System.out.println("Testing DArray mat mat mult batched method...");

        int aRows = 2;
        int aColsBRows = 2;
        int bCols = 2;
        int batchCount = 2;

        double timesAB = 1.0;
        double timesResult = 0.0;

        int lda = aRows;
        int ldb = aColsBRows;
        int ldResult = aRows;

        int strideA = aRows * aColsBRows;
        int strideB = aColsBRows * bCols;
        int strideResult = aRows * bCols;

        DArray matA = Matrix.identity(aRows, handle).asVector().append(Matrix.identity(aRows, handle).asVector()).dArray();
        DArray matB = Matrix.identity(bCols, handle).asVector().append(Matrix.identity(aRows, handle).asVector()).dArray();
        DArray result = DArray.empty(aRows * bCols + aRows * bCols);

        DArray expected = Matrix.identity(aRows, handle).asVector().append(Matrix.identity(aRows, handle).asVector()).dArray();

        DArray2d.multMatMatBatched(
                handle, false, false,
                aRows, aColsBRows, bCols,
                timesAB,
                matA, lda, strideA,
                matB, ldb, strideB,
                timesResult, result, ldResult, strideResult,
                batchCount
        );

        boolean passed = arraysEqual(expected.get(handle), result.get(handle), 1e-4);
        System.out.println("Test passed: " + passed);
        return passed;

    }

    public static boolean testEigen() {

        System.out.println("Testing DArray eigen batched method...");

        Matrix m = new Matrix(handle, 3, 6);
        m.dArray().set(handle, new double[]{
            // First matrix (symmetric)
            4.0, 1.0, 1.0,
            1.0, 3.0, 0.0,
            1.0, 0.0, 2.0,
            // Second matrix (symmetric)
            2.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 3.0
        });

        Vector resultValues = new Vector(handle, m.getWidth());

        System.out.println(m.toString());
        System.out.println(resultValues.toString());
        
        EigenSupport es = new EigenSupport(handle, m, resultValues, 2);
        es.compute(m, resultValues);

        double[] computedEigenvalues = resultValues.toArray();

        // Expected eigenvalues for the given input matrices (manually calculated or from reference library)
        double[] expectedEigenvalues = {
            // Eigenvalues for the first matrix
            5.372, 3.0, 0.628,
            // Eigenvalues for the second matrix
            3.414, 1.0, 1.586
        };

        boolean pass = arraysEqual(computedEigenvalues, expectedEigenvalues, 1e-2);

        System.out.println("Test passed: " + pass);

        return pass;
    }

}
