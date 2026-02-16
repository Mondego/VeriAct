/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

public class AssertionFailureException extends RuntimeException {
  private static final long serialVersionUID = 6988776634359056514L;
  
  //@ modifies this.*;
  AssertionFailureException() { }
  
  //@ modifies this.*;
  AssertionFailureException(String msg) { super(msg); }
}
