public class ScreenPoint {

  private /*@ spec_public @*/ int x, y;
  //@ public invariant 0 <= x && 0 <= y;

  //@ requires 0 <= cs[0] && 0 <= cs[1];
  //@ assignable x, y;
  //@ ensures x == cs[0] && y == cs[1];
  public ScreenPoint(int[] cs)
  { x = cs[0]; y = cs[1]; }

  //@ requires 0 <= j;
  //@ requires y+j < Integer.MAX_VALUE;
  //@ assignable y;
  //@ ensures y == \old(y+j);
  public void moveUp(int j)
  { y = y + j; }

  //@ requires 0 <= i;
  //@ requires x+i < Integer.MAX_VALUE;
  //@ assignable x;
  //@ ensures x == \old(x+i);
  public void moveRight(int i)
  { x = x + i; }

  //@ requires 0 <= i && 0 <= j;
  //@ requires x+i < Integer.MAX_VALUE;
  //@ requires y+j < Integer.MAX_VALUE;
  //@ assignable x, y;
  //@ ensures x == \old(x+i);
  //@ ensures y == \old(y+j);
  public void move(int i, int j) {
    moveRight(i);
    //@ assert x == \old(x+i);
    moveUp(j);
    //@ assert y == \old(y+j);
    //@ assert x == \old(x+i); // ??
  }
}
