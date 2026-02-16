/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

/**
 * This exception is used by <code>Assert</code> to signal that an
 * unimplemented feature has been encountered.  It should only be
 * created by <code>Assert</code>, but may be caught by anyone.
 *
 * @see Assert
 */

public class NotImplementedException extends RuntimeException {
  private static final long serialVersionUID = 582846680511107743L;
  
  //@ normal_behavior
  //@ requires s != null;
  //@ modifies this.*;
  /*package*/ NotImplementedException(String s) {
    super(s);
  }
}
