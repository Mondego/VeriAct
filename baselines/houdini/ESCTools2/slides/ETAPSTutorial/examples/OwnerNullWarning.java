public class OwnerNullWarning {
  boolean allNull = true;
  /*@ non_null */ Object[] a = new Object[10];

  public OwnerNullWarning() {
    //@ set owner = this;
  }
}
