package main;


import algebra.Matrix;
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
        
        Vector v2 = new Vector(dArray, 2, handle);
        
        ArrayList<Boolean> tests = new ArrayList<>();
        
        
        v2 = v2.append(v2);
        
//        tests.add(v2.equals(new Vector(handle, 1,3,5,1,3,5)));
        
        System.out.println(v2);
        
        
        
        
        
        
        
        
        
        
        
        System.out.println("failed tests " + 
                Arrays.toString(IntStream.range(0, tests.size()).filter(i -> !tests.get(i)).toArray())
        );
        
        System.out.println("passed tests " + 
                Arrays.toString(IntStream.range(0, tests.size()).filter(i -> tests.get(i)).toArray())
        );
    }


}
