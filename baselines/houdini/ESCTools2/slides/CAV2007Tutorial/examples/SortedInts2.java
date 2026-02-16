class SortedInts2 {
  private /*@ spec_public rep @*/ int[] a;

  /*@ public invariant (\forall int i, j; 
    @      0 <= i && i < j && j < a.length;
    @      a[i] <= a[j]);     @*/

  /*@ requires 0 < a.length;
    @ ensures \result == a[0];
    @ ensures (\forall int i, j; 
    @      0 <= i && i < a.length;
    @      \result <= a[i]);     @*/
  public /*@ pure @*/ int first()
  { return a[0]; }

  /*@ requires (\forall int i, j; 
    @    0 <= i && i < j && j < inp.length;
    @    inp[i] <= inp[j]);
    @ assignable a;
    @ ensures \fresh(a);
    @ ensures a.length == inp.length;
    @ ensures (\forall int i;
    @      0 <= i && i < inp.length;
    @      a[i] == inp[i]);         @*/
  public SortedInts2(int[] inp) {
    a = new /*@ rep @*/ int[inp.length];
    for (int i = 0; i < a.length; i++) {
        a[i] = inp[i];
  } }
}
