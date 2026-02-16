public class NonNull {
  /*@ non_null */ Object o;
  public void m(/*@ non_null */ Object oo) { o = oo; } 
}
