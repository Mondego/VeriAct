public class UninitWarning {
  public void m() {
    /*@ uninitialized */ int i = 0;
    int j = i; // warning
  }
}
