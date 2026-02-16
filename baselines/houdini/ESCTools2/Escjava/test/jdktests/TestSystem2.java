//#FLAGS: -Q


public class TestSystem2 extends LocalTestCase {


/* FIXME - need to write this test
  public int temp;

  public void testSecurityManager() {
    SecurityManager s = System.getSecurityManager();
    temp = 0;
    SecurityManager ss = System.getSecurityManager();
    assertTrue( s == ss);
    assertTrue( s.equals(ss));
        
   // FIXME - test get, set SecurityManager

  }
*/

  public void testTime() {
    long l = System.currentTimeMillis();
    wasteTime();
    long ll = System.currentTimeMillis();
    assertTrue( ll >= l);
    long lll = System.currentTimeMillis();
    assertTrue( lll >= ll);
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void wasteTime() {
    int k = 0;
    for (int i=0; i<10000000; i++) k += i;
  }

  public void testHashCode() {
    Object o = new Object();
    int i = System.identityHashCode(o);
    Object oo = new Object();
    int ii = System.identityHashCode(oo);
    assertTrue ( oo.hashCode() == ii);
    assertTrue ( o.hashCode() == i);
    //assertTrue (i != ii);
    ii = System.identityHashCode(null);
    assertTrue (ii == 0);
    //@ assert false; // TEST FOR CONSISTENCY
  }

}
