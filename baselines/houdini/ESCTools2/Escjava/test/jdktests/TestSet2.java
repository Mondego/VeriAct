import java.util.*;

public class TestSet2 extends LocalTestCase {
  static E e = new E();
  static F f = new F();

  //@ ensures \result != null && \fresh(\result);
  //@ ensures \result.isEmpty();
  //@ ensures \result.containsNull;
  //@ ensures \result.elementType == \type(E);
  public Set newSet() {
    Set s = new HashSet();
    //@ set s.containsNull = true;
    //@ set s.elementType = \type(E);
    return s;
  }

  public void testAdd() {
    //@ assume e != null;
    //@ assume f != null;
    Set s = newSet();

    //@ assert \typeof(e) <: s.elementType;
    assertTrue( !s.contains(null));
    assertTrue( !s.contains(e));
    assertTrue( !s.contains(f));


    s.add(null);
    //@ assert \typeof(e) <: s.elementType;
    s.add(e);
    assertTrue( s.contains(null));
    assertTrue( s.contains(e));
    //@ assert (\forall Object o; s.containsObject(o); o == e);
    //@ assume !e.equals(f);
    assertTrue( !s.contains(f));

    E ee = new E();
    //@ assert (\forall Object o; s.containsObject(o); o == e);
    //@ assume ee != null;
    //@ assume !e.equals(ee);
    //@ assert !s.containsObject(ee);
    assertTrue( !s.contains(ee));
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testT() {
    //@ assume e != null;
    //@ assume f != null;
    Set s = newSet();

    s.add(null);
    s.add(e);
    //@ assert (\forall Object o; s.containsObject(o); o == e);
//@ assert false; // TEST FOR CONSISTENCY
  }

}

class E {
    //@ also
    //@ public normal_behavior
    //@   ensures \result <==> (this == o);
    //@ pure
    public boolean equals(Object o) { return this == o; }

}


class F {}
