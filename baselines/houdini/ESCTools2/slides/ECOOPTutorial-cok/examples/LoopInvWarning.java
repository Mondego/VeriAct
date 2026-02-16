public class LoopInvWarning {
  public int max(/*@ non_null */ int[] a) {
    int m=Integer.MAX_VALUE;
    //@ loop_invariant (\forall int j; 0<=j && j<i; a[j] <= m);
    //@ decreases a.length - i - 1;
    for (int i=0; i<a.length; ++i) {
      if (m < a[i]) m = a[i];
    }
    return m;
  }
}
