public class Counter2 {

  private /*@ spec_public @*/ int val;

  //@ requires this != y;
  //@ assignable val;
  //@ ensures val == \old(val + y.val);
  //@ ensures y.val == \old(y.val);
  public void addInto(Counter2 y)
  { val += y.val; }
}
