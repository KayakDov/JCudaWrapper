package algebra;

import array.DArray;
import array.Kernel;
import java.io.File;
import java.util.Arrays;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import resourceManagement.Handle;

/**
 * Computes Eigen values for a batch symmetric 2x2 and 3x3 matrices. The
 * matrices should be ordered horizontally and consecutively. Only upper
 * triangles will be considered.
 *Take a llook at https://docs.nvidia.com/cuda/cusolver/index.html?highlight=cusolverDnCheevjBatched#dense-eigenvalue-solver-reference-legacy.
 * It might be faster.
 * @author E. Dov Neimand
 */
public class Eigen {

    private final Kernel sqrt;
    private Kernel cos, acos;
    private final int batchSize;
    private Vector ones3; //to help with eigen3x3

    /**
     * Constructs a machine that computes eigenvalues and eigenvectors on the
     * given batch size.
     *
     * @param batchSize The number of matrices in a batch.
     */
    public Eigen(int batchSize) {
        this.sqrt = new Kernel("sqrt.ptx", "sqrtKernel", batchSize);
        this.batchSize = batchSize;
    }

    /**
     * Computes the eigenvalues for a set of symmetric 2x2 matrices.
     *
     * Data in the bottom triangle will be overwritten.
     *
     * @param all2x2Matrices The set of symmetric 2x2 matrices. There should be
     * at least batch size elements in here. The matrices should have a column
     * dist of 1. Only the upper right triangle is considered.
     * @param vals The eigenvalues will overwrite whatever is here.
     */
    public void computeVals2x2(Vector all2x2Matrices, Vector vals) {
        Vector a = all2x2Matrices.getSubVector(0, batchSize, 4);
        Vector trace = all2x2Matrices.getSubVector(1, batchSize, 4);
        Vector c = all2x2Matrices.getSubVector(2, batchSize, 4);
        Vector d = all2x2Matrices.getSubVector(3, batchSize, 4);

        trace.set(d);
        trace.addToMe(1, a);//= a + d

        Vector valsEven = vals.getSubVector(0, batchSize, 2);
        Vector valsOdd = vals.getSubVector(1, batchSize, 2);

        valsEven.mapEbeMultiplyToSelf(trace, trace); //= (d + a)*(d + a)

        valsOdd.mapEbeMultiplyToSelf(c, c);
        valsOdd.mapAddEbeMultiplyToSelf(a, d, -1);// = ad - c^2

        valsEven.addToMe(-4, valsOdd);//=(d + a)^2 - 4(ad - c^2)
        sqrt.mapToSelf(valsEven);//sqrt((d + a)^2 - 4(ad - c^2))

        valsOdd.set(trace);
        valsOdd.addToMe(-1, valsEven);

        valsEven.addToMe(1, trace);

        vals.mapMultiplyToSelf(0.5);
    }

    /**
     * Computes the eigenvalues for a set of symmetric 3x3 matrices.
     *
     * Data in the bottom triangle will be overwritten.
     *
     * @param all3x3Matrices The set of symmetric 3x3 matrices. There should be
     * at least batch size elements in here. The matrices should have a column
     * dist of 1. Only the upper right triangle is considered.
     * @param vals The eigenvalues will overwrite whatever is here.
     */
    public void computeVals3x3(Vector all3x3Matrices, Vector vals) {
        Vector[][] m = new Vector[3][3];
        Arrays.setAll(m[0], i -> all3x3Matrices.getSubVector(i, batchSize, 9));
        Arrays.setAll(m[1], i -> all3x3Matrices.getSubVector(i + 3, batchSize, 9));
        Arrays.setAll(m[2], i -> all3x3Matrices.getSubVector(i + 6, batchSize, 9));

        //m := a, d, g, d, e, h, g, h, i = m00, m10, m20, m01, m11, m21, m02, m12, m22
        //p := tr m = a + e + i
        //q := (p^2 - norm(m)^2)/2 where norm = a^2 + d^2 + g^2 + d^2 + ...
        // solve: lambda^3 - p lambda^2 + q lambda - det m        
        Vector diag1 = all3x3Matrices.getSubVector(0, 3, 4);

        Vector trace = m[1][0];
        if (ones3 == null)
            ones3 = new Vector(all3x3Matrices.getHandle(), 3).fill(1);
        trace.setBatchVecVecMult(diag1, 9, ones3, 0);

        Vector[][] minor = new Vector[3][3];

        setDiagonalMinors(minor, m, vals);
        Vector C = m[2][0].fill(0);
        for (int i = 0; i < 3; i++) C.addToMe(1, minor[i][i]);

        setRow0Minors(minor, m, vals);
        Vector det = m[2][1];
        det.mapEbeMultiplyToSelf(m[0][0], minor[0][0]);
        det.addEbeMultiplyToSelf(-1, m[0][1], minor[0][1], 1);
        det.addEbeMultiplyToSelf(-1, m[0][2], minor[0][2], -1);

        trace.mapMultiplyToSelf(-1);

        System.out.println("algebra.Eigen.computeVals3x3() coeficiants: " + trace + ", " + C + ", " + det);

        // TODO check solve: lambda^3 - p lambda^2 + q lambda - det m = 0       
        cubicRoots(trace, C, det, vals); // Helper function

    }

    /**
     * Sets the minors of the diagonal elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param workSpace A space where the minors can be stored.
     */
    private void setDiagonalMinors(Vector[][] minor, Vector[][] m, Vector workSpace) {
        for (int i = 0; i < 3; i++)
            minor[i][i] = workSpace.getSubVector(i, batchSize, 3);

        minor[0][0].mapEbeMultiplyToSelf(m[1][1], m[2][2]);
        minor[0][0].addEbeMultiplyToSelf(-1, m[1][2], m[1][2], 1);

        minor[1][1].mapEbeMultiplyToSelf(m[0][0], m[2][2]);
        minor[1][1].addEbeMultiplyToSelf(-1, m[0][2], m[0][2], 1);

        minor[2][2].mapEbeMultiplyToSelf(m[0][0], m[1][1]);
        minor[2][2].addEbeMultiplyToSelf(-1, m[0][1], m[0][1], 1);
    }

    /**
     * Sets the minors of the first row of elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param workSpace A space where the minors can be stored.
     */
    private void setRow0Minors(Vector[][] minor, Vector[][] m, Vector workSpace) {
        for (int i = 0; i < 3; i++)
            minor[0][i] = workSpace.getSubVector(i, batchSize, 3);
        minor[0][1].mapEbeMultiplyToSelf(m[0][1], m[2][2]);
        minor[0][1].addEbeMultiplyToSelf(-1, m[0][2], m[1][2], 1);
        minor[0][2].mapEbeMultiplyToSelf(m[0][1], m[1][2]);
        minor[0][2].addEbeMultiplyToSelf(-1, m[1][1], m[0][2], 1);
    }

    /**
     * Helper method to solve the cubic equation and compute the eigenvalues.
     * notation is taken from:
     * https://en.wikipedia.org/wiki/Cubic_equation#:~:text=A%20cubic%20equation%20with%20real,it%20has%20a%20rational%20root.
     */
    /**
     * Computes the real roots of a cubic equation in the form: x^3 + b x^2 + c
     * x + d = 0
     *
     * TODO: Since this method calls multiple kernels, it would probably be
     * faster if written as a single kernel.
     *
     * @param b Coefficients of the x^2 terms.
     * @param c Coefficients of the x terms.
     * @param d Constant terms.
     * @param roots An array of Vectors where the roots will be stored.
     */
    private void cubicRoots(Vector b, Vector c, Vector d, Vector roots) {
        if (cos == null) {
            cos = new Kernel("cos.ptx", "cosKernel", roots.getDimension());
            acos = new Kernel("acos.ptx", "acosKernel", b.getDimension());
        }

        Vector[] root = new Vector[3];
        Arrays.setAll(root, i -> roots.getSubVector(i, batchSize, 3));

        Vector q = root[0];
        q.mapEbeMultiplyToSelf(b, b);
        q.addEbeMultiplyToSelf(2.0 / 27, q, b, 0);
        q.addEbeMultiplyToSelf(-1.0 / 3, b, c, 1);
        q.addToMe(1, d);

        Vector p = d;
        p.addEbeMultiplyToSelf(1.0 / 9, b, b, 0);
        p.addToMe(-1.0 / 3, c); //This is actually p/-3 from wikipedia.

        //c is free for now.  
        Vector theta = c;
        Vector pInverse = p.mapEBEInverse(root[1]); //c is now taken
        sqrt.map(pInverse, theta);
        theta.addEbeMultiplyToSelf(-0.5, q, theta, 0);//root[0] is now free (all roots).
        theta.mapEbeMultiplyToSelf(theta, pInverse); //c is now free.
        acos.mapToSelf(theta);

        for (int k = 0; k < 3; k++) {
            root[k].set(theta);
            root[k].mapAddToSelf(-2 * Math.PI * k);
        }
        roots.mapMultiplyToSelf(1.0 / 3);
        cos.mapToSelf(roots);

        sqrt.mapToSelf(p);
        for (Vector r : root) {
            r.addEbeMultiplyToSelf(2, p, r, 0);
            r.addToMe(-1.0 / 3, b);
        }
    }

    public void computeVecs2x2(Vector all2x2Matrices, Vector vals) {
        
    }

    public void computeVecs3x3(Vector all3x3Matrices, Vector vals) {
        
        
    }
    
    private getVec(Vector matrices, Vector val, int matrixSize){
        //solve (A - lambdaI)v = 0
        
    }

    public static void main(String[] args) {
        Handle handle = new Handle();
        Matrix m = new Matrix(handle, new double[][]{{1, 2, 3}, {2, 3, 1}, {3, 1, 2}});
        Eigen eigen = new Eigen(1);
        Vector vals = new Vector(handle, 3);
        eigen.computeVals3x3(m.asVector(), vals);
        System.out.println(vals.toString());
    }

}
