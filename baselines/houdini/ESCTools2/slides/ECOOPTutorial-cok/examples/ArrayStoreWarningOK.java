public class ArrayStoreWarningOK {
  public void m(Object o) {
    Object[] s = new String[10];
    if (o instanceof String) s[0] = o;
  }
}
