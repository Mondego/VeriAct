// This test checks that having an exceptional_behavior spec-case in a pure
// method that is used in an annotation does not introduce inconsistencies.
// Checks that a fixed bug does not get broken.

class P {

    public P defaults;

    /*@ public normal_behavior
      @   requires name != null;
      @   ensures (defaults != null) ==> \result == defaults.getProperty(name);
      @*/
    /*@ also public exceptional_behavior
      @   requires name == null;
      @   signals_only NullPointerException;
      @*/
    /* @ public behavior
          ensures (name != null && defaults != null) ==> \result == defaults.getProperty(name);
          signals (Exception e) (e instanceof NullPointerException) && (name == null || defaults == null);
    */
    //@ pure
    public Object getProperty(Object name) throws RuntimeException;
}

public class EXC {

  public static void main(P p) {
    //@ assume p != null;
    try {
      p.getProperty(null);
      //@ assert false;  // OK - not executed
    } catch (Exception e) {
      //@ assert e instanceof NullPointerException;
    }
    //@ assert false;  // SHOULD FAIL
  }

  public static void m(P p) {
    //@ assume p != null;
    try {
      p.getProperty(null);
      //@ assert false;  // OK - not executed
    } catch (Exception e) {
      //@ assert e instanceof NullPointerException;
    //@ assert false;  // SHOULD FAIL
    }
  }

  public static void mm(P p) {
    //@ assume p != null;
    try {
      p.getProperty(null);
      //@ assert false;  // OK - not executed
    } catch (Exception e) {
      //@ assert e instanceof ArithmeticException; // SHOULD FAIL
    }
  }

}

