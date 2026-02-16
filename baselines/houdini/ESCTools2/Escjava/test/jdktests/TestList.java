
import java.util.*;

public class TestList extends LocalTestCase {

    public void testListt() {
        testList(new LinkedList());
        testListR(new LinkedList());
        testList2(new LinkedList());
        testList3(new LinkedList());
        testList4(new LinkedList());
        testList5(new LinkedList());
        testList6(new LinkedList());
        testList7(new LinkedList());
        testList8(new LinkedList());
    }

    public void testVector() {
       testList(new Vector());
       testListR(new Vector());
       testList2(new Vector());
       testList3(new Vector());
       testList4(new Vector());
       testList5(new Vector());
       testList6(new Vector());
       testList7(new Vector());
       testList8(new Vector());
    }

    public void testArray() {
       testList(new ArrayList());
       testListR(new ArrayList());
       testList2(new ArrayList());
       testList3(new ArrayList());
       testList4(new ArrayList());
       testList5(new ArrayList());
       testList6(new ArrayList());
       testList7(new ArrayList());
       testList8(new ArrayList());
    }

    //@ non_null
    Object o = new Object();

    //@ requires c != null;
    //@ requires c.isEmpty();
    private void testList(List c) {
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
    public void testListR(List c) {
/* FIXME - takes too much time
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
*/
//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testList2(List c) {
        //@ set c.elementType = \type(Number);
        boolean b;
        Integer i = new Integer(0);
        c.add(i);
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
    public void testList2b(List c) {
        //@ set c.elementType = \type(Number);
        boolean b;
        Integer i = new Integer(0);
        c.add(i);
        b = c.contains(i);
        assertTrue (b);       

        try {
	    b = c.retainAll(null); // illegal argument - Throws exception
            assertTrue( false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
	    b = c.contains(i);
	    assertTrue (b);
        }
//@ assert false; // TEST FOR CONSISTENCY - no longer relevant due to use of illegal argument in call to retainAll()
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testList3(List c) {
        //@ set c.elementType = \type(Number);
        boolean b;
        Integer i = new Integer(0);
        c.add(i);

        try {
	    b = c.removeAll(null); // illegal argument - Throws exception
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

//@ assert false; // TEST FOR CONSISTENCY - no longer relevant due to use of illegal argument in call to removeAll()
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    public void testList4(List c) {
        //@ set c.elementType = \type(Number);

        Object[] a;
        a = c.toArray();
        assertTrue ( a.getClass() == Object[].class);
        assertTrue ( a.length == 0);

        Integer i = new Integer(0);
        c.add(i);

        a = c.toArray();
        assertTrue ( a.getClass() == Object[].class);
        assertTrue ( a.length == 1);
        //@ assert Arrays.contains(a,i);

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
    //@ requires c.elementType == \type(Object);
    public void testList5(List c) {
        //@ set c.elementType = \type(Object);

        c.add(o);
        assertTrue( !c.isEmpty());

        c.clear();
        assertTrue( c.isEmpty());
        assertTrue ( c.size() == 0);
        assertTrue (!c.contains(o));
//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    //@ requires c.elementType == \type(Object);
    public void testList6(List c) {
        c.add(o);
        assertTrue( c.size() == 1);
        try {
           c.get(1);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
        try {
           c.get(-1);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
        try {
           c.set(1,o);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
        try {
           c.set(-1,o);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
        try {
           c.remove(1);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
        try {
           c.remove(-1);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
        try {
           c.add(-1,o);
           assertTrue (false);
        } catch (Exception e) {
           assertTrue( e instanceof RuntimeException);
        }
        try {
           c.add(3,o);
        } catch (Exception e) {
           assertTrue( e instanceof IndexOutOfBoundsException);
        }
           //@ assert c.elementType == \type(Object);
        try {
        assertTrue( c.size() == 1);
        c.set(0,o);
        assertTrue( c.size() == 1);
        c.remove(0);
        assertTrue( c.size() == 0);
        } catch (RuntimeException e) {
           //@ assert c.elementType == \type(Object);  // FIXME - this should be ok
           //@ set c.elementType = \type(Object);
        }
        try {
           c.add(0,o);
        } catch (IndexOutOfBoundsException e) {
           assertTrue( false );
        } catch (RuntimeException e) {
        }

//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    //@ requires c.elementType == \type(Object);
    public void testList7(List c) {
        Integer i = new Integer(0);
        Integer ii = new Integer(1);
        Integer iii = new Integer(2);
        try {
        c.add(i);
        assertTrue( c.get(0) == i);
        c.add(0,ii);
        assertTrue( c.get(0) == ii);
        assertTrue( c.get(1) == i);
        c.set(0,iii);
        assertTrue( c.get(0) == iii);
        assertTrue( c.get(1) == i);
        c.remove(0);
        assertTrue( c.get(0) == i);
        } catch (RuntimeException e) {}
//@ assert false; // TEST FOR CONSISTENCY
    }

    //@ requires c != null;
    //@ requires c.isEmpty();
    //@ requires c.elementType == \type(Object);
    public void testList8(List c) {
/* FIXME - takes too long
        Integer i = new Integer(0);
        Integer ii = new Integer(1);
        Integer iii = new Integer(2);
        c.add(i);
        c.add(ii);
        c.add(i);
        //@ assert !i.equals(ii);
        //@ assert !ii.equals(iii);
        //@ assert !i.equals(iii);
        //@ assert c.size() == 3;
        //@ assert Collection.nullequals(i,c.get(0));
        int k = c.indexOf(i);
        //@ assert c.contains(i);
        //@ assert Collection.nullequals(i,c.get(0));
        //@ assert k == 0;
        assertTrue (c.indexOf(i) == 0); // FIXME - see List.indexOf
        assertTrue (c.indexOf(ii) == 1); // FIXME - see List.indexOf
        assertTrue (c.indexOf(iii) == -1);
        assertTrue (c.lastIndexOf(i) == 2);
        assertTrue (c.lastIndexOf(ii) == 1);
        assertTrue (c.lastIndexOf(iii) == -1);
*/
//@ assert false; // TEST FOR CONSISTENCY
    }

// FIXME - need other tests with addAll, retainAll, removeAll

        // FIXME - need to test add, remove with containsNull, elementType

        // FIXME = need to test toArray (both), with containsNull, elementType

        // FIXME - Need to test iterator(), listIterator()

        



}
