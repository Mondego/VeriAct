public class ScreenPoint2 {

  private /*@ spec_public @*/ int x, y;
  //@ public invariant 0 <= x && 0 <= y;

  //@ requires 2 <= cs.length;
  //@ requires 0 <= cs[0] && 0 <= cs[1];
  //@ assignable x, y;
  //@ ensures x == cs[0] && y == cs[1];
  public ScreenPoint2(int[] cs)
  { x = cs[0]; y = cs[1]; }
}
