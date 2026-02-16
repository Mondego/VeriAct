// Tests some functionality unique to ArrayList

import java.util.ArrayList;

public class TestArrayList2 extends LocalTestCase {

  public void testCapacity() {
    ArrayList a = new ArrayList(10);
    //@ assert a.capacity == 10;
    Integer i = new Integer(1);
    a.add(i);
    assertTrue( a.size() == 1 );
    //@ assert a.get(0) == i;
    a.ensureCapacity(20);
    //@ assert a.get(0) == i;
    assertTrue( a.size() == 1 );
    //@ assert a.capacity >= 20;
    //@ assert a.containsNull;
    //@ assert a.elementType == \type(Object);
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testTrim() {
    ArrayList a = new ArrayList(10);
    //@ set a.containsNull = false;
    //@ set a.elementType = \type(Number);
    a.add(new Integer(1));
    //@ assert a.size() == 1;
    a.trimToSize();
    //@ assert a.capacity == 1;
    //@ assert a.size() == 1;
    //@ assert !a.containsNull;
    //@ assert a.elementType == \type(Number);
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testClone() {
    ArrayList a = new ArrayList();
    assertTrue ( a.getClass() == ArrayList.class );
    //@ assert \typeof(a) == \type(ArrayList);

    //@ set a.containsNull = false;
    //@ set a.elementType = \type(Number);
    Integer i = new Integer(1);
    a.add(i);
    ArrayList b = (ArrayList)a.clone();
    //@ assert b.size() == 1;
    //@ assert !b.containsNull;
    //@ assert b.elementType == \type(Number);
    //@ assert b.equals(a);
    //@ assert a.get(0) == i;
    //@ assert b.get(0) == i;
    //@ assert \typeof(b) == \type(ArrayList);
    //@ assert false; // TEST FOR CONSISTENCY
  }
}
