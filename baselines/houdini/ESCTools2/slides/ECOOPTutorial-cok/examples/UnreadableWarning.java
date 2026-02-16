public class UnreadableWarning {
  //@ readable_if b; // OLD (non-JML) syntax
  //@ writable_if b;
  public int i; //@ readable i if b; // New syntax
  public int j;
  public boolean b;

  void m() {
    i = j;
    if (b) j = i;
    j = i;
  }
}
