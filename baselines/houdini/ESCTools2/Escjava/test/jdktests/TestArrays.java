import java.util.*;

public class TestArrays extends LocalTestCase {
// FIXME - needs tests for fill with restricted range; contains with restricted range, binarySearch, sort with restricted range
// FIXME - should we test floating point?
// FIXME - test sorts for Objects

  public void testBinSearchInt() {
    int[] a = new int[10];
    for (int i=0; i<a.length; ++i) a[i] = 2*i;
    int k = Arrays.binarySearch(a,8);
    assertTrue( k == 4);
    k = Arrays.binarySearch(a,7);
    assertTrue( k == -5);
  }

  int k1,k2,k3;

  public void NoExectestBinSearchInt2() {
    //@ assume a != null;
    //@ assume a.length == 10;
    //@ assume (\forall int i,j; 0<=i && i<j && j<a.length; a[i] <= a[j]);
    int key = a[5];
    int k = Arrays.binarySearch(a,key);
    assertTrue( key == a[k]);
/* FIXME - too much time
    //@ assume k1 < a[0];
    //@ assert( Arrays.binarySearch(a,k1) == -1);
    //@ assume a[9] < k2;
    //@ assert( Arrays.binarySearch(a,k2) == -10);
    //@ assume a[4] < k3 && k3 < a[5];
    //@ assert( Arrays.binarySearch(a,k3) == -6);
*/
  }

  public void testBinSearchInt3() {
    int[] aa = null;
    try {
       int k = Arrays.binarySearch(aa,0);
System.out.println("K " + k);
       assertTrue(false);
    } catch (Exception e) {
       assertTrue( e instanceof NullPointerException );
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testSortInt() {
    //@ assume a != null;
    Arrays.sort(a);
    //@ assert a[3] <= a[7];
    //@ assert a[6] < a[5]; // ERROR
  }

  public void testSortIntN() {
    //@ assume a == null;
    try {
      Arrays.sort(a);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }


    int[] a = new int[10];
    int[] c = new int[10];
    int[] d = new int[10];
    int[] b = new int[9];
    int[] n = null;

  public void testEqualsInt() {
    //@ assume a != null && b != null && c != null && d != null && n == null;
    //@ assume a.length == 10;
    //@ assume b.length == 9;
    //@ assume c.length == 10;
    //@ assume d.length == 10;
    //@ assume (\forall int i; 0<=i && i <10; a[i] == i);
    //@ assume (\forall int i; 0<=i && i <9; b[i] == i);
    //@ assume (\forall int i; 0<=i && i <10; c[i] == i);
    //@ assume (\forall int i; 0<=i && i <9; d[i] == i);
    //@ assert Arrays.equals(n,n);
    //@ assert Arrays.equals(n,null);
    //@ assert !Arrays.equals(a,null);
    //@ assert Arrays.equals(a,a);
    //@ assert !Arrays.equals(a,b);
    //@ assert Arrays.equals(a,c);
    //@ assert Arrays.equals(a,d); // ERROR - FIXME
  }

    Object[] oa = new Object[10];
    Object[] oc = new Object[10];
    Object[] od = new Object[10];
    Object[] ob = new Object[9];
    Object[] on = null;

  public void testEqualsObjectId() {
    //@ assume oa != null && ob != null && oc != null && od != null && on == null;
    Object o = new Object();
    //@ assume oa.length == 10;
    //@ assume ob.length == 9;
    //@ assume oc.length == 10;
    //@ assume od.length == 10;
    //@ assume (\forall int i; 0<=i && i <10; oa[i] == o);
    //@ assume (\forall int i; 0<=i && i <9; ob[i] == o);
    //@ assume (\forall int i; 0<=i && i <10; oc[i] == o);
    //@ assume (\forall int i; 0<=i && i <9; od[i] == o);
    //@ assert Arrays.equalElements(on,on);
    //@ assert Arrays.equalElements(on,null);
    //@ assert !Arrays.equalElements(oa,null);
    //@ assert Arrays.equalElements(oa,oa);
    //@ assert !Arrays.equalElements(oa,ob);
    //@ assert Arrays.equalElements(oa,oc);
    //@ assert Arrays.equalElements(oa,od); // ERROR - FIXME
  }

// FIXME - actually does not test the use of Object.equals
  public void testEqualsObject() {
    //@ assume oa != null && ob != null && oc != null && od != null && on == null;
    Object o = new Object();
    //@ assume oa.length == 10;
    //@ assume ob.length == 9;
    //@ assume oc.length == 10;
    //@ assume od.length == 10;
    //@ assume (\forall int i; 0<=i && i <10; oa[i] == o);
    //@ assume (\forall int i; 0<=i && i <9; ob[i] == o);
    //@ assume (\forall int i; 0<=i && i <10; oc[i] == o);
    //@ assume (\forall int i; 0<=i && i <9; od[i] == o);
    //@ assert Arrays.equals(on,on);
    //@ assert Arrays.equals(on,null);
    //@ assert !Arrays.equals(oa,null);
    //@ assert Arrays.equals(oa,oa);
    //@ assert !Arrays.equals(oa,ob);
    //@ assert Arrays.equals(oa,oc);
    //@ assert Arrays.equals(oa,od); // ERROR
  }

  public void testContainsInt() {
    int[] a = new int[10];
    a[0] = 0;
    a[1] = 1;
    //@ assert Arrays.contains(a,0); // FIXME
    //@ assert Arrays.contains(a,1);
    //@ assert ! Arrays.contains(a,2);

    // @ assert false; // TEST FOR CONSISTENCY
  }

  public void testContainsObject() {
    Object[] a = new Object[10];
    Object o = new Object();
    Object oo = new Object();
    Object ooo = new Object();
    a[0] = o;
    a[1] = oo;
    //@ assert Arrays.contains(a,o); // FIXME
    //@ assert Arrays.contains(a,oo);
    //@ assert ! Arrays.contains(a,ooo);

    // @ assert false; // TEST FOR CONSISTENCY
  }

  public void testFillInt() {
    int[] a = new int[10];
    Arrays.fill(a,8);
    assertTrue( a[6] == 8);
    a = null;
    try {
      Arrays.fill(a,0);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testFillObject() {
    Object o = new Object();
    Object[] a = new Object[10];
    try {
    Arrays.fill(a,o);
    assertTrue( a[6] == o);
    Arrays.fill(a,null);
    assertTrue( a[6] == null);
    } catch (Exception e) { assertTrue(false); }

    
    a = new Integer[10];
    try {
      Arrays.fill(a,o);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof ArrayStoreException);
    }

    try {
      Arrays.fill(null,o);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testAsList() {
    Object o = new Object();
    Object oo = new Object();
    Object[] a = new Object[10];
    a[0] = o;
    a[1] = oo;
    List list = Arrays.asList(a);
    assertTrue (list.size() == a.length);
    assertTrue (list.get(0) == a[0]);
    assertTrue (list.get(1) == a[1]);
    assertTrue (list.get(0) == o);
    a[0] = oo;
    //assertTrue (list.get(0) == oo); // FIXME - need to record the array as backing store

    try {
      list = Arrays.asList(null);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue ( e instanceof NullPointerException );
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }
}
