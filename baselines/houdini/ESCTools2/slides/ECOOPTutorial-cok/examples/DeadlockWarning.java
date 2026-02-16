public class DeadlockWarning {
  /*@ non_null */ final static Object o = new Object();
  /*@ non_null */ final static Object oo = new Object();

  //@ axiom o < oo;

  //@ requires \max(\lockset) < o;
  public void m() {
      synchronized(o) { synchronized(oo) { }}
  }

  //@ requires \max(\lockset) < o;
  public void mm() {
      synchronized(oo) { synchronized(o) { }} // Deadlock warning
  }
}


