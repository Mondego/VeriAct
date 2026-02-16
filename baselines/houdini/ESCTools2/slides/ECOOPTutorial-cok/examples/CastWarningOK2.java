public class CastWarningOK2 {
  //@ requires o instanceof String;
  public void m(Object o) {
    String s = (String)o;
  }
}
