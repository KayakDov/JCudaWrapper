package algebra;

import algebra.MatrixBatchPntrs;
import algebra.Vector;
import array.DArray;
import array.DBatchArray;
import array.DPointerArray;
import array.IArray;
import array.KernelManager;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.math3.exception.DimensionMismatchException;
import resourceManagement.Handle;

/**
 * This class provides methods for handling a batch of strided matrices stored
 * in row-major format. Each matrix in the batch can be accessed and operated on
 * individually or as part of the batch. The class extends {@code Matrix} and
 * supports operations such as matrix multiplication and addition. The strided
 * storage is designed to support JCuda cuSolver methods that require row-major
 * memory layout.
 *
 * Strided matrices are stored with a defined distance (stride) between the
 * first elements of each matrix in the batch.
 *
 * @author E. Dov Neimand
 */
public class MatrixStride extends Matrix {

    private int subWidth;
    private DBatchArray batchArray;

    /**
     * Constructor for creating a batch of strided matrices. Each matrix is
     * stored with a specified stride and the batch is represented as a single
     * contiguous block in memory.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param subWidth The number of columns (width) in each submatrix.
     * @param stride The number of elements between the first elements of
     * consecutive submatrices in the batch.
     * @param batchSize The number of matrices in this batch.
     */
    public MatrixStride(Handle handle, int subHeight, int subWidth, int stride, int batchSize) {
        super(handle, subHeight, subWidth * batchSize);
        batchArray = data.getStrided(stride);
        this.subWidth = subWidth;
    }
    
    /**
     *  Creates a simple batch matrix with coldDist = height.
     * @param handle
     * @param data The length of this data should be width*height.
     * @param height The height of this matrix.
     */
    public MatrixStride(Handle handle, DBatchArray data, int height){
        super(handle, data, height, data.length/height);
        this.batchArray = data;
        this.subWidth = data.stride/height;
    }
    

    /**
     * Constructor for creating a batch of square strided matrices. Each matrix
     * is stored with a specified stride and the batch is represented as a
     * single contiguous block in memory. These matrices have no overlap.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatrixStride(Handle handle, int subHeight, int batchSize) {
        this(handle, subHeight, subHeight, batchSize);
    }

    /**
     * Constructor for creating a batch of square strided matrices. Each matrix
     * is stored with a specified stride and the batch is represented as a
     * single contiguous block in memory. These matrices have no overlap.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param subWidth The number of columns (width) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatrixStride(Handle handle, int subHeight, int subWidth, int batchSize) {
        this(handle, subHeight, subHeight, subHeight * subWidth, batchSize);
    }

    /**
     * Returns a vector of elements corresponding to row {@code i} and column
     * {@code j} across all matrices in the batch. Each element in the vector
     * corresponds to the element at position (i, j) in a different submatrix.
     *
     * @param i The row index in each submatrix.
     * @param j The column index in each submatrix.
     * @return A vector containing the elements at (i, j) for each submatrix in
     * the batch.
     */
    public Vector get(int i, int j) {
        return asVector().getSubVector(index(i, j), batchArray.batchCount(), batchArray.stride);
    }

    /**
     * Retrieves all elements from each submatrix in the batch as a 2D array of
     * {@code Vector} objects. Each vector contains elements corresponding to
     * the same row and column across all matrices in the batch. The method
     * returns a 2D array where each element (i, j) is a vector of the elements
     * at position (i, j) from each matrix in the batch. "i" is the row and "j"
     * is the column.
     *
     * @return A 2D array of {@code Vector} objects. Each {@code Vector[i][j]}
     * represents the elements at row {@code i} and column {@code j} for all
     * submatrices in the batch.
     */
    public Vector[][] getAll() {
        Vector[][] all = new Vector[getSubHeight()][getSubHeight()];
        for (int i = 0; i < getSubHeight(); i++) {
            int row = i;
            Arrays.setAll(all[row], col -> get(row, col));
        }
        return all;
    }

    /**
     * Returns the number of columns (width) in each submatrix.
     *
     * @return The width of each submatrix in the batch.
     */
    public int getSubWidth() {
        return subWidth;
    }

    /**
     * Returns the number of rows (height) in each submatrix.
     *
     * @return The height of each submatrix in the batch.
     */
    public int getSubHeight() {
        return getHeight();
    }

    /**
     * Performs matrix multiplication and on the batches of matrices, and add
     * them to this matrix. This method multiplies matrix batches {@code a} and
     * {@code b}, scales the result by {@code timesAB}, scales the existing
     * matrix in the current instance by {@code timesResult}, and then adds them
     * together and palces the result here.
     *
     * @param transposeA Whether to transpose the matrices in {@code a}.
     * @param transposeB Whether to transpose the matrices in {@code b}.
     * @param a The left-hand matrix batch in the multiplication.
     * @param b The right-hand matrix batch in the multiplication.
     * @param timesAB The scaling factor applied to the matrix product
     * {@code a * b}.
     * @param timesResult The scaling factor applied to the result matrix.
     * @return this
     * @throws DimensionMismatchException if the dimensions of matrices
     * {@code a} and {@code b} are incompatible for multiplication.
     */
    public MatrixStride multAndAdd(boolean transposeA, boolean transposeB, MatrixStride a, MatrixStride b, double timesAB, double timesResult) {
        if (a.getSubWidth() != b.getSubHeight())
            throw new DimensionMismatchException(a.getSubWidth(),
                    b.getSubHeight());

        // Perform batched matrix multiplication and addition
        batchArray.multMatMatStridedBatched(getHandle(), transposeA, transposeB,
                a.getSubHeight(), a.getSubWidth(), b.getSubWidth(),
                timesAB,
                a.batchArray, a.colDist,
                b.batchArray, b.colDist,
                timesResult, colDist);
        
        return this;
    }

    /**
     * Computes the eigenvalues. This batch must be a set of symmetric 2x2
     * matrices.
     *
     * @param workSpace Should be at least as long as batchSize.
     * @return The eigenvalues.
     */
    public Vector computeVals2x2(Vector workSpace) {

        Vector vals = new Vector(getHandle(), getWidth());
        Vector[] val = vals.parition(2);

        Vector[][] m = getAll();

        Vector trace = workSpace.getSubVector(0, batchArray.batchCount());

        trace.set(m[1][1]);
        trace.addToMe(1, m[0][0]);//= a + d

        val[0].mapEbeMultiplyToSelf(trace, trace); //= (d + a)*(d + a)

        val[1].mapEbeMultiplyToSelf(m[0][1], m[0][1]); //c^2
        val[1].mapAddEbeMultiplyToSelf(m[0][0], m[1][1], -1);// = ad - c^2

        val[0].addToMe(-4, val[1]);//=(d + a)^2 - 4(ad - c^2)
        KernelManager.get("sqrt").mapToSelf(getHandle(), val[0]);//sqrt((d + a)^2 - 4(ad - c^2))

        val[1].set(trace);
        val[1].addToMe(-1, val[0]);
        val[0].addToMe(1, trace);

        vals.mapMultiplyToSelf(0.5);

        return vals;
    }



    /**
     * Computes the eigenvalues for a set of symmetric 3x3 matrices. If this
     * batch is not such a set then this method should not be called.
     *
     * @param workSpace Should have length equal to the width of this vector.
     * @return The eigenvalues.
     *
     */
    //m := a, d, g, d, e, h, g, h, i = m00, m10, m20, m01, m11, m21, m02, m12, m22
    //p := tr m = a + e + i
    //q := (p^2 - norm(m)^2)/2 where norm = a^2 + d^2 + g^2 + d^2 + ...
    // solve: lambda^3 - p lambda^2 + q lambda - det m        
    public Vector computeVals3x3(Vector workSpace) {

        Vector vals = new Vector(handle, getWidth());
        Vector[] work = workSpace.parition(3);

        Vector[][] m = getAll();

        Vector[][] minor = new Vector[3][3];

        Vector negTrace = negativeTrace(work[0], work[1].getSubVector(0,
                getHeight()));//val[0] is taken, but val[1] is free.

        setDiagonalMinors(minor, m, vals);
        Vector C = work[1].fill(0);
        for (int i = 0; i < 3; i++) C.addToMe(1, minor[i][i]);

        setRow0Minors(minor, m, vals);
        Vector det = work[2];
        det.mapEbeMultiplyToSelf(m[0][0], minor[0][0]);
        det.addEbeMultiplyToSelf(-1, m[0][1], minor[0][1], 1);
        det.addEbeMultiplyToSelf(-1, m[0][2], minor[0][2], -1);

//        System.out.println("algebra.Eigen.computeVals3x3() coeficiants: " + trace + ", " + C + ", " + det);
        cubicRoots(negTrace, C, det, vals); // Helper function

        return vals;
    }

    /**
     * The negative of the trace of the submatrices.
     *
     * @param traceStorage The vector that gets overwritten with the trace.
     * Should have batch elements.
     * @param ones a vector that will have -1's stored in it. It should have
     * height number of elements in it.
     * @return The trace.
     */
    private Vector negativeTrace(Vector traceStorage, Vector ones) {//TODO:Retest!
        
        ones.fill(-1);
        
        VectorStride diagnols = new VectorStride(handle, traceStorage.dArray().getStrided(9), 3, 4);
        
        Vector trace = new VectorStride(handle, traceStorage, 1, 1);
        
        trace.setBatchVecVecMult(diagnols, new VectorStride(handle, trace, 0, 3));
        
        return trace;
    }

    /**
     * Sets the minors of the diagonal elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param minorStorage A space where the minors can be stored.
     */
    private void setDiagonalMinors(Vector[][] minor, Vector[][] m, Vector minorStorage) {
        minor[0] = minorStorage.parition(subWidth);

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
     * @param minorStorage A space where the minors can be stored.
     */
    private void setRow0Minors(Vector[][] minor, Vector[][] m, Vector minorStorage) {
        minor[0] = minorStorage.parition(getSubWidth());

        minor[0][1].mapEbeMultiplyToSelf(m[1][1], m[2][2]);
        minor[0][1].addEbeMultiplyToSelf(-1, m[1][2], m[1][2], 1);

        minor[0][1].mapEbeMultiplyToSelf(m[0][1], m[2][2]);
        minor[0][1].addEbeMultiplyToSelf(-1, m[0][2], m[1][2], 1);

        minor[0][2].mapEbeMultiplyToSelf(m[0][1], m[1][2]);
        minor[0][2].addEbeMultiplyToSelf(-1, m[1][1], m[0][2], 1);
    }

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
    private static void cubicRoots(Vector b, Vector c, Vector d, Vector roots) {
        KernelManager cos = KernelManager.get("cos"),
                acos = KernelManager.get("acos"),
                sqrt = KernelManager.get("sqrt");

        Vector[] root = roots.parition(3);

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
        sqrt.map(b.getHandle(), pInverse, theta);
        theta.addEbeMultiplyToSelf(-0.5, q, theta, 0);//root[0] is now free (all roots).
        theta.mapEbeMultiplyToSelf(theta, pInverse); //c is now free.
        acos.mapToSelf(b.getHandle(), theta);

        for (int k = 0; k < 3; k++) {
            root[k].set(theta);
            root[k].mapAddToSelf(-2 * Math.PI * k);
        }
        roots.mapMultiplyToSelf(1.0 / 3);
        cos.mapToSelf(b.getHandle(), roots);

        sqrt.mapToSelf(b.getHandle(), p);
        for (Vector r : root) {
            r.addEbeMultiplyToSelf(2, p, r, 0);
            r.addToMe(-1.0 / 3, b);
        }
    }

    /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite. TODO: If any extra memory is available, pass
     * it here!
     *
     * @param eValues The eigenvalues, organized by sets per matrix.
     * @param eVectors Where the eigenvectors will be placed, organized by sets
     * per matrix. This should be an empty matrix the same dimensions as this.
     * @param info Status of operation success.
     */
    private void computeVecs(Vector eValues, MatrixStride eVectors) {

        try (
                MatrixBatchPntrs pointersB = new MatrixBatchPntrs(eVectors,
                        getSubHeight(), getSubWidth(), getHeight(), 1);
                IArray pivotArray = IArray.empty(getHeight() * batchArray.batchCount());
                IArray info = IArray.empty(batchArray.batchCount());
                
                MatrixStride wrkspForCopyOfThis = new MatrixStride(handle, getSubHeight(), subWidth, batchArray.stride, batchArray.batchCount());
                
                DPointerArray ptrsWorkSpcThis = DPointerArray.empty(batchArray.batchCount(),getSubWidth() * getSubHeight())
            ) {

            eVectors.fill(0);

            Vector[] value = eValues.parition(getHeight());

            for (int i = 0; i < getHeight(); i++) {
                computeVec(value[i], pointersB, info, pivotArray,
                        wrkspForCopyOfThis, ptrsWorkSpcThis);
                pointersB.shiftPointers(handle, 1, 0);
            }
        }

    }

    /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite.
     *
     * TODO: this method currently solves with LU factorization. There may be a
     * faster method that uses Cholesky decomposition since this method will be
     * used on positive definite matrices. However, A - lambda v is not positive
     * definite, even if A is, so a more complex approach is required.
     *
     * @param values The eigenvalues.
     * @param b Where the eigenvector will be placed.
     * @param info The success of the computations.
     */
    private void computeVec(Vector values, MatrixBatchPntrs b, IArray info, IArray pivot, MatrixStride auxMatrWorkSpaceCopyA, DPointerArray pntrsWorkSpaceA) {

        copy(auxMatrWorkSpaceCopyA);

        for (int i = 0; i < getHeight(); i++)
            auxMatrWorkSpaceCopyA.get(i, i).addToMe(-1, values);

        MatrixBatchPntrs ptrsToCopyA = auxMatrWorkSpaceCopyA.getPointers(
                pntrsWorkSpaceA);//Do not close.  data stored in pntrsWorkSpaceA

        System.out.println(
                "algebra.MatrixBatchStride.computeVec() needs LU factoring:\n" + auxMatrWorkSpaceCopyA.toString());

        ptrsToCopyA.LUFactor(handle, pivot, info);

        System.out.println(
                "algebra.MatrixBatchStride.computeVec() is LU Factored:\n" + auxMatrWorkSpaceCopyA.toString());
        System.out.println(
                "algebra.MatrixBatchStride.computeVec() with pivoting:\n" + pivot.toString());
        System.out.println(
                "algebra.MatrixBatchStride.computeVec() and info:\n" + info.toString());

        ptrsToCopyA.solveLUFactored(handle, b, pivot, info);

    }

    /**
     * Returns this matrix as a set of pointers.
     *
     * @return
     */
    public MatrixBatchPntrs getPointers() {

        return new MatrixBatchPntrs(getSubHeight(), getSubWidth(), colDist, batchArray.getPointerArray(handle));
                
    }

    /**
     * Returns this matrix as a set of pointers.
     *
     * @param putPointersHere An array where the pointers will be stored.
     * @return
     */
    public MatrixBatchPntrs getPointers(DPointerArray putPointersHere) {
        return new MatrixBatchPntrs(
                getSubHeight(), getSubWidth(), colDist, putPointersHere.fill(handle, batchArray)
        );

    }

    @Override
    public void close() {
        super.close();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < batchArray.batchCount(); i++)
            sb.append(getSubMatrix(0, getHeight(), i * getSubWidth(),
                    (i + 1) * getSubWidth())).append("\n");
        return sb.toString();
    }

    @Override
    public MatrixStride copy() {
        MatrixStride copy = new MatrixStride(handle, getSubHeight(),
                subWidth, batchArray.stride, batchArray.batchCount());
        return copy(copy);
    }

    /**
     * Copies from this matrix into the proffered matrix.
     *
     * @param copyTo becomes a copy of this matrix.
     * @return the copy.
     */
    public MatrixStride copy(MatrixStride copyTo) {
        if (colDist == getHeight())
            copyTo.dArray().set(handle, dArray(), 0, 0, getHeight() * getWidth());
        else copyTo.addAndSet(1, this, 0, this);
        return copyTo;
    }

    public static void main(String[] args) {
        try (
                Handle handle = new Handle();
                MatrixStride mbs = new MatrixStride(handle, 2, 2);
                Vector workSpace = new Vector(handle, 4)) {

            mbs.dArray().set(handle, new double[]{4, 2, 2, 3, 4, 1, 1, 3});

            try (Vector eigenVals = mbs.computeVals2x2(workSpace)) {

                System.out.println(eigenVals);
                MatrixStride eigenVecs = new MatrixStride(handle, 2, 2);
                mbs.computeVecs(eigenVals, eigenVecs);
                System.out.println(eigenVecs.toString());
            }

        }
    }

}
