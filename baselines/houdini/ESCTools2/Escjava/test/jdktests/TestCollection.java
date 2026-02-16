
import java.util.*;

public class TestCollection extends LocalTestCase {

    public void testList() throws RuntimeException {
        testCollection(new LinkedList());
        testCollectionR(new LinkedList());
        testCollection2(new LinkedList());
        testCollection3(new LinkedList());
        testCollection4(new LinkedList());
        testCollection5(new LinkedList());
    }

    public void testSet() throws RuntimeException {
       testCollection(new HashSet());
       testCollectionR(new HashSet());
       testCollection2(new HashSet());
       testCollection3(new HashSet());
       testCollection4(new HashSet());
       testCollection5(new HashSet());
    }

    //public void testBag() throws RuntimeException {
//       testCollection(new MultiSet());
    //}

    //@ non_null
    Object o = new Object();

    //@ requires c != null;
    //@ requires c.isEmpty();
    private void testCollection(Collection c) throws RuntimeException {
        boolean b;

        assertTrue( c.isEmpty());
        assertTrue (c.size() == 0);
        assertTrue (!c.contains(o));
        assertTrue (!c.contains(null));
        //@ assert c.content.theSize == 0;

        //@ set c.elementType = \type(Number);
        Integer i = new Integer(0);

        //@ assert c.content.theSize == 0;
        //@ assume !i.equals(o);
        assertTrueNP (!c.contains(null));
        //@ assert !c.containsObject(null);
        assertTrueNP (!c.contains(o));
        //@ assert !c.containsObject(o);

        //@ assert c.content.theSize == 0;
        //@ assert !c.contains(i);
        b = c.add( i );
        assertTrue (b);
        assertTrue ( !c.isEmpty());
        assertTrue ( c.size() == 1);
        assertTrueNP ( c.contains(i));
        //@ assert c.containsObject(i);
        assertTrueNP ( !c.contains(o));
        //@ assert !c.containsObject(o);

        Integer ii = new Integer(1);
        //@ assert !ii.equals(i);
        //@ assume !c.contains(ii); // FIXME - would like to prove this
        b = c.add(ii);
        assertTrue(b);
        //@ assert c.containsObject(i);
        //@ assert c.containsObject(ii);
        //@ assert !c.containsObject(o);
//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testCollectionR(Collection c) {
        try {
        //@ set c.elementType = \type(Number);
        //@ assert (\forall Object o; !c.contains(o) );

        boolean b;
        Integer i = new Integer(0);
        //@ assume !i.equals(o);
        c.add(i);
        //@ assert (\forall Object o; c.contains(o); o.equals(i) );

        Integer ii = new Integer(1);
        //@ assert !ii.equals(i);
        //@ assert !c.containsObject(ii);
        //@ assume !c.contains(ii); // FIXME - would like to prove this
        b = c.add(ii);
        //@ assume !ii.equals(o);
        assertTrue(b);
        //@ assert c.containsObject(i);
        //@ assert c.containsObject(ii);
        //@ assert !c.containsObject(o);

        //@ assume (\forall Object o; c.contains(o); o.equals(i) || o.equals(ii)); // FIXME - would like to prove this
        b = c.remove(ii);
        assertTrue(b);
        //@ assert !c.containsObject(ii);
        //@ assert c.containsObject(i);

        b = c.remove(ii);
        assertTrue(!b);
        //@ assert c.containsObject(i);
        //@ assert !c.containsObject(ii);

        try {
	    b = c.containsAll(null);
            assertTrue( false);
        } catch (NullPointerException e) {
            assertTrue( e instanceof NullPointerException);
        }

        //@ assert c != null;
        b = c.containsAll(c);
        assertTrue (b);

        try {
	    b = c.addAll(null);
            assertTrue( false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
        b = c.contains(i);
        assertTrue (b);
//@ assert false; // TEST FOR CONSISTENCY
        } catch (Exception e) {}
//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testCollection2(Collection c) throws RuntimeException {
        //@ set c.elementType = \type(Number);
        boolean b;
        Integer i = new Integer(0);
        c.add(i);

        try {
	    b = c.retainAll(null);
            assertTrue( false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
        b = c.contains(i);
        assertTrue (b);

        try {
	    b = c.retainAll(c);
	    assertTrue(!b);
	    b = c.contains(i);
	    assertTrue (b);
        } catch (Exception e) {}

//@ assert false; // TEST FOR CONSISTENCY

    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testCollection3(Collection c) throws RuntimeException {
        //@ set c.elementType = \type(Number);
        boolean b;
        Integer i = new Integer(0);
        c.add(i);

        try {
	    b = c.removeAll(null);
            assertTrue( false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
        b = c.contains(i);
        assertTrue (b);

        try {
	    b = c.removeAll(c);
            assertTrue(true);
            assertTrue(b);
            assertTrue( !c.contains(i));
            assertTrue( c.isEmpty());
        } catch (Exception e) {
        }
//@ assert false; // TEST FOR CONSISTENCY

    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testCollection4(Collection c) {
        //@ set c.elementType = \type(Number);

        Object[] a;
        // a = c.toArray();
        // assertTrue ( a.getClass() == Object[].class);
        // assertTrue ( a.length == 0);

        Integer i = new Integer(0);
        c.add(i);

        // a = c.toArray();
        // assertTrue ( a.getClass() == Object[].class);
        // assertTrue ( a.length == 1);
        // //@ assert Collection.contain(a,i);

        a = new Object[10];
        //@ assert a != null;
        //@ assert c.elementType <: \elemtype(\typeof(a));
        Object[] b;
	try {
        b = c.toArray(a);
        assertTrue( a== b);
        assertTrue( a[2] == null);
        assertTrue( b.length == 10);
        } catch (NullPointerException e) {
           //@ assert false;
	} catch (ArrayStoreException e) {
           //@ assert false;
        } catch (Exception e) {
           //@ assert false;
        }

        try {
	    a = c.toArray(null);
            assertTrue( false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testCollection5(Collection c) throws RuntimeException {
        //@ set c.elementType = \type(Object);

        c.add(o);
        assertTrue( !c.isEmpty());

        c.clear();
        assertTrue( c.isEmpty());
        assertTrue ( c.size() == 0);
        assertTrue (!c.contains(o));
//@ assert false; // TEST FOR CONSISTENCY
    }
// FIXME - need other tests with addAll, retainAll, removeAll

        // FIXME - need to test add, remove with containsNull, elementType

        // FIXME = need to test toArray (both), with containsNull, elementType

        // FIXME - Need to test iterator()

        // FIXME - need to test clone, equals
        



}
