public class NullWarningOK {
  public void m(/*@ non_null */ Object o) {
    int i = o.hashCode();
  }
}
