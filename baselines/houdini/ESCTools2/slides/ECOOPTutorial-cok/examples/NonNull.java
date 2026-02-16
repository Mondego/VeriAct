public class NonNull {
  /*@ non_null */ Object o;

  public void m(Object oo) { o = oo; } // NonNull warning
}
