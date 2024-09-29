package unmodifiable;

import array.DSingleton;
import jcuda.driver.CUdeviceptr;
import resourceManagement.Handle;

/**
 * An unmodifiable dArray.
 *
 * @author E. Dov Neimand
 */
public class DArray extends array.DArray {

    /**
     * This exception is thrown if an attempt to modify the matrix is made.
     */
    public static class UnmodifiableException extends UnsupportedOperationException {

        public UnmodifiableException() {
            super("This matrix is unmodifiable.");
        }

    }

    /**
     * A shallow copy of the array.
     *
     * @param p The array's pointer.
     * @param length The array's length.
     */
    public DArray(CUdeviceptr p, int length) {
        super(p, length);
    }

    @Override
    public void set(Handle handle, array.DArray from, int toStart, int fromStart, int toInc, int fromInc, int length) {
        throw new UnmodifiableException();
    }

    @Override
    public void set(Handle handle, double[] from, int toIndex) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray subArray(int start, int length) {
        return super.subArray(start, length).unmodifiable();
    }

    @Override
    public DSingleton get(int index) {
        throw new UnmodifiableException();
    }

    public unmodifiable.DSingleton getSingleton(int index) {
        return new unmodifiable.DSingleton(pointer);
    }

    @Override
    public int matrixAddWithTranspose(Handle handle, boolean transA, boolean transB, int heightA, int widthA, double alpha, array.DArray a, int lda, double beta, array.DArray b, int ldb, int ldc) {
        throw new UnmodifiableException();
    }

    @Override
    public void outerProd(Handle handle, int rows, int cols, double multProd, array.DArray vecX, int incX, array.DArray vecY, int incY, int lda) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray multMatVec(Handle handle, boolean transA, int aRows, int aCols, double timesAx, array.DArray matA, int lda, array.DArray vecX, int incX, double beta, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray multBandMatVec(Handle handle, boolean transposeA, int rowsA, int colsA, int subDiagonalsA, int superDiagonalA, double timesA, array.DArray Ma, int ldm, array.DArray x, int incX, double timesThis, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray solveTriangularBandedSystem(Handle handle, boolean isUpper, boolean transposeA, boolean onesOnDiagonal, int rowsA, int nonPrimaryDiagonals, array.DArray Ma, int ldm, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray multSymBandMatVec(Handle handle, boolean upper, int colA, int diagonals, double timesA, array.DArray Ma, int ldm, array.DArray x, int incX, double timesThis, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray fill(Handle handle, double fill, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public void multMatMat(Handle handle, boolean transposeA, boolean transposeB, int aRows, int bCols, int aCols, double timesAB, array.DArray a, int lda, array.DArray b, int ldb, double timesCurrent, int ldc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray addToMe(Handle handle, double timesX, array.DArray x, int incX, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray addAndSet(Handle handle, boolean transA, boolean transB, int height, int width, double alpha, array.DArray a, int lda, double beta, array.DArray b, int ldb, int ldc) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray multMe(Handle handle, double mult, int inc) {
        throw new UnmodifiableException();
    }

    @Override
    public void matrixSquared(Handle handle, boolean transpose, int uplo, int resultRowsCols, int cols, double alpha, array.DArray a, int lda, double timesThis, int ldThis) {
        throw new UnmodifiableException();
    }

    @Override
    public array.DArray atan2(array.DArray from) {
        throw new UnmodifiableException();
    }

    /**
     * This method has been disabled. Be sure to call close from the creator
     * instance.
     */
    @Override
    public void close() {
        throw new UnmodifiableException();
    }

    @Override
    public DArray fill0(Handle handle) {
        throw new UnmodifiableException();
    }

}
