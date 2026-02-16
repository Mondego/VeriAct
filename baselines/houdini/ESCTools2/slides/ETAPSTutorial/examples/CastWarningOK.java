public class CastWarningOK {
  public void m(Object o) {
    if (o instanceof String) { String s = (String)o; }
  }
}
