public class ArrayStoreWarning {
  public void m(Object o) {
    Object[] s = new String[10];
    s[0] = o;
  }
}
