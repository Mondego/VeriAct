/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

/**
 * This checked exception is used to signal a dynamic "type error" in
 * the use of S-expressions.
 *
 * <p> <em>E.g.</em>, it is thrown if a non-integer is treated as an
 * integer, or an attempt is made to reference a non-existent element
 * of a list. </p>
 */

final public class SExpTypeError extends Exception {
  private static final long serialVersionUID = 5767797092888960711L;
}
