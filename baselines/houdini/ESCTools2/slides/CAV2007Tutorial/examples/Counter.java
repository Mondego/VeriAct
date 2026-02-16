public class Counter {

  private /*@ spec_public @*/ int val;

  //@ assignable val;
  //@ ensures val == \old(val + y.val);
  //@ ensures y.val == \old(y.val);
  public void addInto(Counter y)
  { val += y.val; }
}
