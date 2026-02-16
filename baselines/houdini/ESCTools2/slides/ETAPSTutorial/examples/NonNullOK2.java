public class NonNull {
  /*@ non_null */ Object o;
  public void m(Object oo) { 
    if (oo != null) o = oo; 
  }
}
