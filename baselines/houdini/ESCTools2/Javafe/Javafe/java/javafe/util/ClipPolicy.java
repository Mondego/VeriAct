/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

/**
 * Objects of this class can decide on various aspects of the clipping
 * policy of lines displayed to the user with a caret.
 */

public class ClipPolicy
{
  /**
   * @return <code>true</code> if the programming construct that
   * begins at column <code>pos</code> within <code>s</code> also ends
   * within <code>s</code>.
   */

  //@ requires 0 <= pos && pos < s.length();
  public boolean containsEndOfConstruct(/*@ non_null */ String s, int pos) {
    return true;
  }
}

