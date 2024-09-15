package main;

import algebra.Matrix;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;
import processSupport.Handle;
import storage.DArray;
import storage.DSingleton;

/**
 *
 * @author edov
 */
public class MatrixTest {

    public static void main(String[] args) {
        double[][] sixCount2d = {{1, 2}, {3, 4}, {5, 6}};

        Handle hand = new Handle();

        Matrix A = new Matrix(hand, sixCount2d);
                
        System.out.println("A = \n" + A.toString() + "\n");
        
        DArray sixCount1D = new DArray(hand, 1,2,9,3,4,9,5,6,9);
        
        Matrix B = new Matrix(sixCount1D, 2, 3, 3, hand).transpose();
                
        
        System.out.println("B = \n" + B.toString() + "\n");        
        
        ArrayList<Boolean> tests = new ArrayList<>();
        
        tests.add(A.getHeight() == 2); //0
        tests.add(A.getWidth() == 3); //1
        tests.add(A.equals(B.transpose()));//2
        tests.add(A.multiply(B).equals(new Matrix(new DArray(hand, 35, 44, 44, 56), 2, 2, hand)));//3
        tests.add(A.add(B.transpose()).equals(A.multiply(2)));//4
        tests.add(A.subtract(A).equals(new Matrix(2, 3, hand).fill(0)));//5
        tests.add(A.multiply(1).equals(A));//6
        
        Matrix addOneResult = new Matrix(
                sixCount1D.addToMe(hand, 1, new DSingleton(1, hand), 0, 1), 
                        2, 3, 3, hand);
        
        tests.add(A.scalarAdd(1).equals(addOneResult));//7
        tests.add(A.insert(new Matrix(hand, new double[][]{{100}}), 0, 2)
                .equals(new Matrix(hand, new double[][]{{1, 2}, {3, 4}, {100, 6}})));//8
        
        tests.add(A.getEntry(1, 1) == 4.0);//9
        
        double[][] subMat = new double[2][2];
        
        B.copySubMatrix(1, 3, 0, 2, subMat);
        
        tests.add(Arrays.deepEquals(subMat, new double[][]{{3, 5},{4, 6}}));//10
        
        tests.add(B.getSubMatrix(new int[]{1}, new int[]{1}).getEntry(0,0) == 4);//11
        
        B.setSubMatrix(new double[][]{{99}}, 2, 0);
        
        tests.add(B.equals(new Matrix(hand, new double[][]{{1, 3, 99}, {2, 4, 6}})));//12
        
        
        tests.add(B.size() ==6);//13       
        
        tests.add(B.copy().equals(B)); //14
        
        B.setEntry(2, 0, 5);
        
        A.setEntry(0, 2, 5);
        
        tests.add(B.transpose().equals(A));//15
        
        tests.add(B.getRowMatrix(1).equals(new Matrix(hand, new double[][]{{3},{4}})));//16
        
        tests.add(Arrays.equals(B.getRow(1), new double[]{3,4}));//17
        
        tests.add(Arrays.equals(B.getColumn(1), new double[]{2,4,6}));//18
                
        tests.add(Arrays.deepEquals(A.getData(), sixCount2d));//19
        
        Matrix square = new Matrix(hand, new double[][]{{1,2},{3,4}});
        
        tests.add(square.getTrace() == 5);//20
                
        tests.add(Arrays.equals(B.operate(new double[]{1,1}), new double[]{3,7,11}));//21
        
        tests.add(Matrix.identity(2, hand).equals(new Matrix(hand, new double[][]{{1,0},{0,1}})));//22
        
        tests.add(square.power(2).equals(new Matrix(hand, new double[][]{{7, 10},{15, 22}})));//23
        
        A.setColumn(1, new double[]{1, 2});
        
        tests.add(Arrays.equals(A.getColumn(1), new double[]{1, 2}));//24
        
        A.setColumnMatrix(1, Matrix.fromColVec(new double[]{7,8}, hand));//25
        
        tests.add(Arrays.equals(A.getColumn(1), new double[]{7, 8}));//26
        
        A.setRow(1, new double[]{3, 2, 1});
        
        tests.add(Arrays.equals(A.getRow(1), new double[]{3, 2, 1}));//27
        
        System.out.println(B);
        
        
        System.out.println("failed tests " + 
                Arrays.toString(IntStream.range(0, tests.size()).filter(i -> !tests.get(i)).toArray())
        );
        
        System.out.println("passed tests " + 
                Arrays.toString(IntStream.range(0, tests.size()).filter(i -> tests.get(i)).toArray())
        );
        
    }

}
