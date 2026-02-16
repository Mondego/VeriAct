import java.util.Vector;

public class TestVector extends LocalTestCase {

    public void testVtc1() {
        Vector v = new Vector();
        //@ assert false; // TEST FOR CONSISTENCY 1
    }

    public void testVtc2() {
        Vector v = new Vector(3);
        //@ assert false; // TEST FOR CONSISTENCY 2
    }


    public void testVt3() {
        Vector v = new Vector(3);
        v.add(this);
        //@ assert v.elementCount == 10; // SHOULD FAIL
    }


    public void testVt4() {
        Vector v = new Vector(3);
        //@ assert v.isEmpty();
        v.add(this);
        //@ assert v.elementCount == 1; // SHOULD PASS

        v.add(this);
        //@ assert v.elementCount == 2; // SHOULD PASS

        v.add(this);
        //@ assert v.elementCount == 3; // SHOULD PASS
    }


    //@ requires arr != null;
    public void testVtR_positive(Object[] arr) {
        Vector v = new Vector(arr.length);

        //@ loop_invariant i >= 0 & i <= arr.length;
        //@ loop_invariant v.elementType == \type(Object);
        //@ loop_invariant v.containsNull;
        for (int i = 0; i < arr.length; i++) {
            v.add(arr[i]);
        }

        v.removeAllElements();
        //@ assert v.isEmpty(); // SHOULD PASS
    }

}

