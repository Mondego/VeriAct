// This test was a bug in that uses of pure methods in loop invariants
// did not result in the axioms for the pure method being added to the
// verification condition; introducing the redundant assert statement
// triggered the addition of the axioms, so the proof would succeed.
class Allpos {
      Allpos();

      //@ public normal_behavior
      //@ requires f!=null && n<=f.length;
      //@ ensures \result == (\forall int i; 0<=i & i<n; f[i]>0);
      //@ pure
      //@ model static boolean allpos(int[] f, int n);

      //@ public normal_behavior
      //@   requires f != null;
      //@   ensures \result == f.length - k;
      //@ pure
      //@ model static int diff(int[] f, int k);

      //@ requires f!=null;
      //@ ensures \result == (\forall int i; 0<=i & i<f.length; f[i]>0);
      boolean allposLoop(int[] f) {
         int  k = 0;
         boolean res = true;

         // assert res == allpos(f, k);     // NEEDED
         //@ loop_invariant 0<=k && k<=f.length;
         //@ loop_invariant res == allpos(f, k);
         //@ decreases diff(f,k);
         while (k != f.length) {
             res = res && f[k]>0;
             k = k+1;
         }
         return res;
      }

      Object o;

      //@ requires f!=null;
      //@ ensures \result == (\forall int i; 0<=i & i<f.length; f[i]>0);
      boolean allposLoop2(int[] f) {
         int  k = 0;
         boolean res = true;

         // assert res == allpos(f, k);     // NEEDED
         //@ loop_invariant 0<=k && k<=f.length;
         //@ loop_invariant res == allpos(f, k);
         //@ decreases diff(f,k);
         while (k != f.length) {
             res = res && f[k]>0;
             o = null ; // force a state change
             k = k+1;
         }
         return res;
      }

}
