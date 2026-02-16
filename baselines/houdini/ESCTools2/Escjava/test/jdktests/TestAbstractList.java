import java.util.*;

public class TestAbstractList extends LocalTestCase {

   public void testRemoveRange( ) {
	Z z = new Z();
        //@ set z.elementType = \type(Object);
	for (int i=0; i<20; ++i) z.add(i,new Integer(i));
	testRemoveRange(z);
	
   }

   //@ requires z != null;
   public void testRemoveRange( Z z ) {
        //@ assume z.theSize == 20;
        //@ assume z.elementType == \type(Object);
        assertTrue (z.size() == 20);
	try {
		z.removeRange(-1,8);
		assertTrue(false);
        } catch (Exception e) {
            assertTrue( e instanceof IndexOutOfBoundsException);
        }
        assertTrue (z.size() == 20);
	try {
		z.removeRange(41,48);
		assertTrue(false);
        } catch (Exception e) {
            assertTrue( e instanceof IndexOutOfBoundsException);
        }
        assertTrue (z.size() == 20);
	try {
		z.removeRange(10,8);
		assertTrue (z.size() == 20);
		//assertTrue(false);
        } catch (Exception e) {
            assertTrue( false );
        }
        assertTrue (z.size() == 20);
/* FIXME - this must use an iterator???
	try {
		z.removeRange(10,31);
		assertTrue (false);
        } catch (Exception e) {
System.out.println("EXC " + e);
            assertTrue( e instanceof IndexOutOfBoundsException);
        }
        assertTrue (z.size() == 10);
*/
	z.removeRange(0,20);
        assertTrue(z.size() == 0);
//@ assert false; // TEST FPOR CONSISTENCY
   }

   static class Z extends AbstractList {
        //@ spec_public non_null
        final private Object[] array = new Object[100]; //@ in objectState;
					//@ maps array[*] \into objectState;
        //@ public invariant array.owner == this;
        //@ public invariant (\forall int i; 0<=i && i<s; array[i] == get(content,i));
        //@ public invariant \elemtype(\typeof(array)) == \type(Object);

        public Z() {
	    //@ set array.owner = this;
        }

        /*@ spec_public @*/ private int s = 0;  //@ in objectState;
        //@ public invariant s <= array.length;
        //@ represents theSize = s;

        public Object get(int i) { 
            if (i<0 || i>=s) throw new IndexOutOfBoundsException();
            return array[i]; }

        public void add(int i, Object o) {
          if (i<0 || i>s) throw new IndexOutOfBoundsException();
          array[i] = o;
          ++s;
        }

        public Object remove(int i) throws IndexOutOfBoundsException {
          if (i<0 || i>=s) throw new IndexOutOfBoundsException();
	  Object o = array[i];
          for (int k=i+1; k<s; ++k) array[k-1] = array[k];
          --s;
	  return o;
        }

        //@ also public normal_behavior
        //@   ensures \result == s;
        //@ pure
        public int size() { return s; }

        public void removeRange(int f, int t) {
		super.removeRange(f,t);
	}
   }
}

