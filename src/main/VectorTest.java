package main;

import algebra.Vector;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;
import processSupport.Handle;
import storage.DArray;

public class VectorTest {

    public static void main(String[] args) {

        Handle handle = new Handle();

        double[] array = {1.0, 2.0, 3.0, 4.0, 5.0, 6};

        Vector v1 = new Vector(handle, array);
        DArray dArray = new DArray(handle, array);
        Vector v2 = new Vector(handle, dArray, 2);

        ArrayList<Boolean> tests = new ArrayList<>();

        // Existing tests
        v2 = v2.append(v2);
        tests.add(v2.equals(new Vector(handle, 1, 3, 5, 1, 3, 5)));//0
        tests.add(v1.getEntry(2) == 3);//1
        v1.setEntry(0, 100);
        tests.add(v1.equals(new Vector(handle, 100, 2, 3, 4, 5, 6)));//2
        v2.close();
        v2 = new Vector(handle, dArray, 4);
        tests.add(v2.getDimension() == 2);//3
        tests.add(v2.addToMe(-1, v2).equals(new Vector(handle, 0, 0)));//4
        tests.add(v1.mapMultiplyToSelf(2).equals(new Vector(handle, 200, 4, 6, 8, 10, 12)));//5
        tests.add(v2.fill(1).equals(new Vector(handle, 1, 1)));//6
        v2 = v1.add(v1.copy().fill(-1));
        tests.add(v2.equals(new Vector(handle, 199, 3, 5, 7, 9, 11)));//7
        tests.add(v2.mapMultiply(0).equals(new Vector(handle, 6).fill(0)));//8
        v2.close();
        v2 = new Vector(handle, 1, 2);
        tests.add(v2.dotProduct(v2) == 5);//9
        tests.add(v1.subtract(v1).equals(new Vector(handle, 6).fill(0)));//10
        tests.add(v2.ebeMultiply(new Vector(handle, 2, 2)).equals(new Vector(handle, 2, 4)));//11
        tests.add(v2.ebeDivide(new Vector(handle, 2, 2)).equals(new Vector(handle, .5, 1)));//12
        tests.add(v1.getSubVector(1, 1).equals(new Vector(handle, new double[]{4})));//13
        v1.setSubVector(0, new Vector(handle, 1, 2));
        tests.add(v1.equals(new Vector(handle, 1, 2, 6, 8, 10, 12)));//14
        tests.add(v2.mapAdd(1).equals(new Vector(handle, 2, 3)));//15
        tests.add(v2.getL1Norm() == 3);//16
        tests.add(v2.getNorm() == Math.sqrt(5));//17
        tests.add(v2.getLInfNorm() == 2);//18
        tests.add(v1.getDistance(new Vector(handle, 1, 2, 6, 8, 10, 10)) == 2.0);//19
        tests.add(v1.getL1Distance(new Vector(handle, 1, 2, 6, 8, 10, 10)) == 2.0);//20
        tests.add(v1.getLInfDistance(new Vector(handle, 1, 2, 6, 8, 10, 10)) == 2.0);//21
        Vector v3 = v1.copy();
        tests.add(v3.equals(v1));//22
        tests.add(v1.mapSubtract(1).equals(new Vector(handle, 0, 1, 5, 7, 9, 11)));//23
        tests.add(v1.mapDivide(2).equals(new Vector(handle, 0.5, 1, 3, 4, 5, 6)));//24
        tests.add(v1.mapSubtractToSelf(1).equals(new Vector(handle, 0, 1, 5, 7, 9, 11)));//25

        // New tests
        // mapToSelf
        tests.add(v1.mapToSelf(Math::sqrt).equals(new Vector(handle, 0, 1, Math.sqrt(5), Math.sqrt(7), Math.sqrt(9), Math.sqrt(11))));//26

        // projection
        Vector v4 = new Vector(handle, 1, 0);
        tests.add(v4.projection(new Vector(handle, 2, 0)).equals(new Vector(handle, 1, 0)));//27

        // unitVector
        Vector v5 = new Vector(handle, 3, 4);
        tests.add(v5.unitVector().equals(new Vector(handle, 0.6, 0.8)));//28

        // unitize
        v5.unitize();
        tests.add(v5.equals(new Vector(handle, 0.6, 0.8)));//29

        // isNaN and isInfinite
        Vector v6 = new Vector(handle, 1, Double.NaN, 2);
        tests.add(v6.isNaN());//30
        tests.add(!v6.isInfinite());//31
        Vector v7 = new Vector(handle, 1, Double.POSITIVE_INFINITY, 2);
        tests.add(!v7.isNaN());//32
        tests.add(v7.isInfinite());//33

        tests.add(v2.cosine(new Vector(handle, 1, 0)) == (1 / Math.sqrt(5))); // 34

        v2.combineToSelf(2, 3, new Vector(handle, 1, 0));

        tests.add(v2.equals(new Vector(handle, 5, 4))); // 35

        v1.mapToSelf(t -> t * t * t * t);

        tests.add(v1.getSubVector(1, 3).equals(new Vector(handle, 1, 25, 49))); // 36

        v1.mapDivideToSelf(2);
        tests.add(v1.equals(new Vector(handle, 0, 0.5, 12.5, 24.5, 40.5, 60.5))); // 37

        tests.add(v1.getMaxValue() == 60.5); // 38

        System.out.println(v1);
        System.out.println(v1.getMinValue());

        tests.add(Math.abs(v1.getMinValue()) < 1e-7); // 39

        tests.add(v1.getMinIndex() == 0); // 40

        tests.add(v1.getMaxIndex() == 5); // 41

        System.out.println("failed tests "
                + Arrays.toString(IntStream.range(0, tests.size()).filter(i -> !tests.get(i)).toArray())
        );

        System.out.println("passed tests "
                + Arrays.toString(IntStream.range(0, tests.size()).filter(i -> tests.get(i)).toArray())
        );
    }
}
