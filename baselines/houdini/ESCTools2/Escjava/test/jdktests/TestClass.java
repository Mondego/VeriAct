// FIXME - there are lots of things not tested.
// I've just done those that are relevant to ESC/Java proofs

import java.util.Vector;
import java.util.List;

public class TestClass extends LocalTestCase {


// getSuperclass, getName, isPrimitive, getComponentType
// isArray, isInstance, isAssignableFrom, forName, toString

  public void testElemtype() {
    // @ assert \elemtype(int.class) == ?;
    // @ assert \elemtype(Object.class) == ?;
    //@ assert \elemtype(int[].class) == \type(int);
    //@ assert \elemtype(Object[].class) == \type(Object);
    //@ assert \elemtype(int[][].class) == \type(int[]);

    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testIsArray() {
/*
    assertTrue( !Object.class.isArray() );
    assertTrue( !Vector.class.isArray() );
    assertTrue( !int.class.isArray() );
    assertTrue( Object[].class.isArray() );
    assertTrue( Vector[].class.isArray() );
    assertTrue( int[].class.isArray() );
    assertTrue( int[][].class.isArray() );
*/
/*
    Object o = new Vector();
    Object oo = new int[10];
    Object ooo = new Vector[20];
    Object o4 = new int[10][20];
    assertTrue ( !o.getClass().isArray() );
    assertTrue ( oo.getClass().isArray() );
    assertTrue ( ooo.getClass().isArray() );
    assertTrue ( o4.getClass().isArray() );
*/
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testIsPrimitive() {
    Object o = new Vector();
    Object oo = new int[10];
    Object ooo = new Vector[20];
    Object o4 = new int[10][20];
    assertTrue ( !o.getClass().isPrimitive() );
    assertTrue ( !oo.getClass().isPrimitive() );
    assertTrue ( !ooo.getClass().isPrimitive() );
    assertTrue ( !o4.getClass().isPrimitive() );

    assertTrue ( !Object.class.isPrimitive());
    assertTrue ( !Object[].class.isPrimitive());
    assertTrue ( !Vector.class.isPrimitive());
    assertTrue ( !Vector[].class.isPrimitive());
    assertTrue ( !int[].class.isPrimitive());
    int i; byte b; boolean bb; short s; long l; float f; double d; char c;
    //@ ghost \bigint I; 
    //@ ghost \real r;
    assertTrue (int.class.isPrimitive());
    assertTrue (byte.class.isPrimitive());
    assertTrue (short.class.isPrimitive());
    assertTrue (long.class.isPrimitive());
    assertTrue (float.class.isPrimitive());
    assertTrue (double.class.isPrimitive());
    assertTrue (char.class.isPrimitive());
    assertTrue (boolean.class.isPrimitive());

    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testIsInstance() {
    Class c = List.class;
    //@ assert c == \type(List);
    Vector v = new Vector();
    String s = new String();
    boolean b = c.isInstance(v);
    assertTrue (b);
    b = c.isInstance(s);
    assertTrue (!b);
    b = c.isInstance(null);
    assertTrue (!b);
    //@ assert false; // TEST FOR CONSISTENCY
  }


}
