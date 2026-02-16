public class ReachableWarning {
  //@ requires i==0 || i ==1;
  public void m(int i) {
    switch (i) {
      case 0: break;
      case 1: break;
      default: //@ unreachable
    }
  }
}
