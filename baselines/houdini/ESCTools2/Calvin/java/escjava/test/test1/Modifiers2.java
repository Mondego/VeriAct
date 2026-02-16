/* Copyright Hewlett-Packard, 2002 */


// Tests for resolution and typechecking of pragmas

public class Modifiers2 extends Modifiers1 {
  int v3;

  public int update(Modifiers1 v1)
    /*@ also_modifies v3; also_ensures v3 == this.v1 */
  {
    v3 = this.v1;
    return super.update(v1);
  }
    
}
