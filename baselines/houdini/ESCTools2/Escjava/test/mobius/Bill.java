package mobius;

/**
 * The Bill class provides an abstract implementation of the bill
 * functionality. It calculates the aggregate cost for series of
 * investments based on the implementation of the method which gives 
 * the cost of a single round (to be implemented in subclasses).
 * 
 * @author Hermann Lehner, Aleksy Schubert, Joseph Kiniry
 */

abstract class Bill
{
  private /*@ spec_public @*/ int sum;
  //@ public invariant 0 <= sum;
 
  /**
   * This method gives a cost of a single round.
   * 
   * @param x is the number of the particular round
   * @return the cost of the investment in this round, it's not
   * greater than <code>x</code>.
   */ 
  //@ ensures 0 <= \result & \result <= x;
  abstract int roundCost(int x) throws Exception;
  
  /**
   * This method calculates the cost of the whole series of
   * investments.
   *
   * @return <code>true</code> when the calculation is successful and
   * <code>false</code> when the calculation cannot be performed.
   */
  //@ requires 0 < n;
  //@ ensures sum <=  \old(sum) + n*(n+1)/2;
  public boolean produce_bill(final int n) {
    int i = 0;
    try {
      //@ loop_invariant 0 < i & i <= n + 1 & 0 <= sum & sum <= \old(sum) + (i - 1) * i / 2;
      for (i = 1; i <= n; i++) {
        sum = sum + roundCost(i);
      }
      return true;
    } catch (Exception e) {
      return false;
    }
  }
}
