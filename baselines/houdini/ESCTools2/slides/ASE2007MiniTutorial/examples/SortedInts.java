class SortedInts {
  private /*@ spec_public @*/ int[] a;

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
    @      0 <= i && i < j && j < inp.length;
    @      inp[i] <= inp[j]);
    @ assignable a;
    @ ensures a == inp;        @*/
  public SortedInts(int[] inp)
  { a = inp; }
}
