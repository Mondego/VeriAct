public class TestObject extends LocalTestCase {

    public void testConst1() {
        Object o = new Object();
        //@ assert o.owner == null;
        Object oo = new Object();
        //@ assert o.owner == null;
//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testClass() {
        Object o = new Object();
        String s = o.toString();
        int i = o.hashCode();
        //@ ghost Object ooo = o.owner;
        Object oo = new Object();
        assertTrue( o.getClass() == Object.class);
        //@ assert o.getClass() == \type(Object);
        //@ assert \typeof(o) == \type(Object);
        //@ assert \typeof(o) == \typeof(oo);
        //@ assert o.getClass() == oo.getClass();
        assertTrue( i == o.hashCode()); // hashCode has not changed
        //@ assert s == o.theString;
        assertTrue( s.equals(o.toString())); // theString has not changed
        //@ assert o.owner == ooo;
        //@ assert o.getClass() == \type(Object);
//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testHashCode() {
        Object o = new Object();
        String s = o.toString();
        int i = o.hashCode();
        //@ ghost Object ooo = o.owner;
        Object oo = new Object();
        int j = o.hashCode() + oo.hashCode();
        assertTrue( i == o.hashCode()); // hashCode has not changed
        assertTrue( s.equals(o.toString())); // theString has not changed
        //@ assert o.owner == ooo;
        //@ assert o.getClass() == \type(Object);
//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testToString() {
        Object o = new Object();
        String s = o.toString();
        int i = o.hashCode();
        //@ ghost Object ooo = o.owner;
        Object oo = new Object();
        String ss = o.toString() + oo.toString();
        assertTrue( i == o.hashCode()); // hashCode has not changed
        assertTrue( s.equals(o.toString())); // theString has not changed
        //@ assert o.owner == ooo;
        //@ assert o.getClass() == \type(Object);
//@ assert false; // TEST FOR CONSISTENCY
    }

    public boolean b; // arbitrary value
    public boolean bb; // arbitrary value

    public void testEquals() {
        Object o = new Object();
        Object oo = new Object();
        Object ooo = b ? o : oo;
        Object oooo = bb ? o : oo;
        assertTrue (!o.equals(null));
        assertTrue (!o.equals(oo));
        assertTrue (o.equals(o));
        assertTrue (o.equals(ooo) == ooo.equals(o));
        assertTrue ( !o.equals(ooo) || (ooo.equals(oooo) == o.equals(oooo)));
//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testEquals2() {
        Object o = new Object();
        String s = o.toString();
        int i = o.hashCode();
        //@ ghost Object ooo = o.owner;

        boolean b = o.equals(o);

        assertTrue( i == o.hashCode()); // hashCode has not changed
        assertTrue( s.equals(o.toString())); // theString has not changed
        //@ assert o.owner == ooo;
        //@ assert o.getClass() == \type(Object);
//@ assert false; // TEST FOR CONSISTENCY
    }

    public void testClone() throws CloneNotSupportedException {
        E e = new E();
        F f = new F();
        Object o = e.clone();
        try {
	    o = f.clone();
        } catch (Exception ex) {
            assertTrue (ex instanceof CloneNotSupportedException);
        }
//@ assert false; // TEST FOR CONSISTENCY
    }

class A implements Cloneable {
    //@ pure
    public A() {}

    //@ also public normal_behavior
    //@    ensures \typeof(\result) <: \type(A);
    //@    ensures \typeof(\result) <: A.class;
    public Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
}

final class E {
    //@ pure
    public E() {}

    //@ also public normal_behavior
    //@    ensures \typeof(\result) == \type(E);
    //@    ensures \typeof(\result) == E.class;
    public Object clone() {
        return new E();
    } // ERROR - must either be a Cloneable or throw a CloneNotSupportedException
}
final class F {
    //@ pure
    public F() {}

    //@ also public exceptional_behavior
    //@    signals_only CloneNotSupportedException;
    public Object clone() throws CloneNotSupportedException {
        throw new CloneNotSupportedException();
    }
}
}
