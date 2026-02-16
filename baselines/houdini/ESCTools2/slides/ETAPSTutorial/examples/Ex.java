public class Ex {
  public void m(Object o) {
    if (!(o instanceof String)) throw new ClassCastException();
  }
}
