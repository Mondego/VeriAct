public class Factor {

  /*@ public behavior
    @   requires 0 <= n;
    @   ensures
    @      n == (\product int i; 
    @             0 <= i && i < \result.length;
    @              \result[i]);
    @   ensures (\forall int i;
    @      0 <= i && i < \result.length;
    @      1 < \result[i] && \result[i] < n
    @      && n % \result[i] == 0);
    @   signals_only PrimeException;
    @   signals (PrimeException e)
    @       (\forall int i; 1 < i && i < n;
    @            n % i != 0);
    @*/
  public /*@ pure @*/static int[] factors(int n)
      throws PrimeException
  { /* a stub */
    return new int[]{1,n};
  }
}
