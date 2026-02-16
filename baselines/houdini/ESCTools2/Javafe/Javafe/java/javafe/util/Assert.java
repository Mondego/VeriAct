/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

/**
 * A class with static methods for checking assertions.
 * (Inspired by <cite>Mastering the AWT</code> by David Geary.)
 */

// FIXME - which of these can we get rid of as no longer used
// FIXME - the notImplemented features in particular should be handled differently
public class Assert {
  //@ ensures false;
  //@ signals_only AssertionFailureException;
  //@ pure
  static public void fail(String msg) {
    throw new AssertionFailureException(msg);
  } //@ nowarn Exception;
  
  //@   requires b;
  //@   ensures true;
  //@   signals (Exception) false;
  //@ also
  //@   requires !b;
  //@   ensures false;
  //@   signals_only AssertionFailureException;
  //@ pure
  static public void notFalse(boolean b) {
    if (!b) throw new AssertionFailureException();
  }
  
  //@   requires b;
  //@   ensures true;
  //@   signals (Exception) false;
  //@ also
  //@   requires !b;
  //@   ensures false;
  //@   signals_only AssertionFailureException;
  //@ pure
  static public void notFalse(boolean b, String msg) {
    if (!b) throw new AssertionFailureException(msg);
  }
  
  //@   requires obj != null;
  //@   ensures true;
  //@   signals (Exception) false;
  //@ also
  //@   requires obj == null;
  //@   ensures false;
  //@   signals_only AssertionFailureException;
  //@ pure
  static public void notNull(Object obj) {
    if (obj == null) throw new AssertionFailureException();
  }
  
  //@   requires obj != null;
  //@   ensures true;
  //@   signals (Exception) false;
  //@ also
  //@   requires obj == null;
  //@   ensures false;
  //@   signals_only AssertionFailureException;
  //@ pure
  static public void notNull(Object obj, String msg) {
    if (obj == null) throw new AssertionFailureException(msg);
  }
  
  //@ ensures false;
  //@ signals_only AssertionFailureException;
  //@ pure
  static public void precondition() {
    throw new AssertionFailureException("Precondition violated!");
  } //@ nowarn Exception;
  
  //@ ensures false;
  //@ signals_only AssertionFailureException;
  //@ pure
  static public void precondition(String msg) {
    throw new AssertionFailureException("Precondition violated: " + msg);
  } //@ nowarn Exception;
  
  //@   requires b;
  //@   ensures true;
  //@   signals (Exception) false;
  //@ also
  //@   requires !b;
  //@   ensures false;
  //@   signals_only AssertionFailureException;
  //@ pure
  static public void precondition(boolean b) {
    if (!b) throw new AssertionFailureException("Precondition violated!");
  }
  
  //@ ensures false;
  //@ signals_only NotImplementedException;
  static public void notImplemented() {
    throw new NotImplementedException("Hit an unimplemented feature");
  }
  
  //@ ensures false;
  //@ signals_only NotImplementedException;
  static public void notImplemented(String s) {
    throw new NotImplementedException("Not implemented: " + s);
  }
}
