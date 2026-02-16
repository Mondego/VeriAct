// This test is prompted by Bug #55

interface A {

  //@ invariant (\typeof(this) == \type(InheritedInvariant)) ==> P();

  //@ normal_behavior pure 
  boolean P();
}

public class InheritedInvariant implements A {

  boolean p;

  //@ normal_behavior
  //@  modifies p;
  //@  ensures p;
  //@  ensures P();
  public InheritedInvariant() {
    p = true;
  }

  //@ also normal_behavior
  //@  ensures \result == p;
  //@ pure
  public boolean P() {
    return p;
  }
}
