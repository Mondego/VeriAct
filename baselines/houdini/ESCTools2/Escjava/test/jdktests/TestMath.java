public class TestMath extends LocalTestCase {

  public void testAbs() {
    //@ assume i != Integer.MIN_VALUE && j != Integer.MIN_VALUE;
    assertTrue ( Math.abs(-2) ==2);
    assertTrue ( Math.abs(i) >= 0);
    assertTrue ( Math.abs(Integer.MIN_VALUE) == Integer.MIN_VALUE);
    assertTrue ( StrictMath.abs(-2) == 2);
    assertTrue ( StrictMath.abs(j) >= 0);
    assertTrue ( StrictMath.abs(Integer.MIN_VALUE) == Integer.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY
  }

  int i=4,j=5;

  public void testMin() {
    assertTrue( Math.min(1,2) == 1);
    assertTrue( Math.min(i,j) <= i );
    assertTrue( StrictMath.min(1,2) == 1);
    assertTrue( StrictMath.min(i,j) <= i );
//@ assert false; // TEST FOR CONSISTENCY
  }
  public void testMax() {
    assertTrue( Math.max(1,2) == 2);
    assertTrue( Math.max(i,j) >= i );
    assertTrue( StrictMath.max(1,2) == 2);
    assertTrue( StrictMath.max(i,j) >= i );
//@ assert false; // TEST FOR CONSISTENCY
  }

// FIXME - needs tests for all the float/double functions

}
