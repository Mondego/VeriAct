/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

/**
 * A <code>FatalError</code> is an unchecked exception thrown only by
 * <code>ErrorSet.fatal</code> that indicates that a fatal error has
 * been encountered, forcing all further processing to be
 * abandoned.  Note that this is really an Exception, not an Error.
 * 
 * <p> The cause and existence of a fatal error will already have been
 * reported to the user by the time the <code>FatalError</code> has
 * been thrown.<p>
 *
 * <p> <code>FatalError</code> must be caught by the top level of the
 * <code>Tool</code> so that any needed cleanup can be done before the
 * <code>Tool</code> exits.  <code>FatalError</code>s should be caught
 * anywhere else only for local cleanup purposes.
 */

public class FatalError extends java.lang.RuntimeException {
  private static final long serialVersionUID = 4916981747846620833L;
  
  /**
   * Create a <code>FatalError</code> exception.  This constructor is
   * intended to be called only by <code>ErrorSet.fatal</code>.
   */
  //@ normal_behavior
  //@ modifies this.*;
  /* package*/ FatalError() {
    super("A fatal Error has occurred");
  }
}
