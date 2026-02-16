class SortedInts3 {
  private /*@ spec_public @*/ int[] a;
  //@ public invariant a.owner == this;

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

  /*@ requires inp.owner != this;
    @ requires (\forall int i, j; 
    @    0 <= i && i < j && j < inp.length;
    @    inp[i] <= inp[j]);
    @ assignable a;
    @ ensures \fresh(a);
    @ ensures a.length == inp.length;
    @ ensures (\forall int i;
    @      0 <= i && i < inp.length;
    @      a[i] == inp[i]);         @*/
  public SortedInts3(int[] inp) {
    a = new int[inp.length]; 
    //@ set a.owner = this;
    for (int i = 0; i < a.length; i++) {
        a[i] = inp[i];
    }
  }
}
