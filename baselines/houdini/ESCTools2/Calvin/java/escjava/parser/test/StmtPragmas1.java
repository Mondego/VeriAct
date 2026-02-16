/* Copyright Hewlett-Packard, 2002 */


// Tests for resolution and typechecking of pragmas

public abstract class StmtPragmas1 {
  static int count;
  static {
    /*@ assume 0 <= count */
    if (count < 0) {
      /*@ unreachable; */
      throw new RuntimeException();
    }
    count++;
    /*@ assert 0 <= count; */
  }

  public static int find(int[] a, int el)
    /*@ requires a != null */
    /*@ requires (\exists int i; 0 <= i & i < a.length & a[i] == el) */
    /*@ requires (\forall int i; 0 <= i-1 & i-1 < a.length ==> a[i-1] < a[i]) */
    /*@ ensures 0 <= \result & \result < a.length & a[\result] == el; */
  {
    int lo = 0, hi = a.length-1;
    if (a[lo] == el) return lo;
    if (a[hi] == el) return hi;
    for(int i = 0; i <= hi; i++) {
      /*@ loop_invariant lo < hi */
      /*@ loop_invariant 0 <= lo & lo < a.length
                       & 0 <= hi & hi < a.length */
      /*@ loop_invariant a[lo] < el & el < a[hi] */
      int mid = lo + (hi - lo)/2;
      if (el == a[mid]) return mid;
      else if (el < a[mid]) hi = mid;
      else lo = mid;
    }
    /*@ unreachable; */
    return -1;
  }
}

