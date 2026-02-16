public class AssertWarning {
  //@ requires i >= 0;
  public void m(int i) {
    //@ assert i >= 0; // OK
    --i;
    //@ assert i >= 0; // FAILS
  }
  public void n(int i) {
    switch (i) {
      case 0,1,2: break;
      default:   //@ unreachable; // FAILS
    }
  }
}
