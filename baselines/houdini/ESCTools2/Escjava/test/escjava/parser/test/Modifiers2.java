
// Tests for resolution and typechecking of pragmas

public class Modifiers2 extends Modifiers1 {
  int v3;

  public int update(Modifiers1 v1)
    /*@ modifies v3; ensures v3 == this.v1 */
  {
    v3 = this.v1;
    return super.update(v1);
  }
    
}
