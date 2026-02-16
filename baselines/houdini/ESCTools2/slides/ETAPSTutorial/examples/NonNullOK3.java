public class NonNull {
  /*@ non_null */ Object o;
  public void m() { 
    o = new Object(); 
  }
}
