/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;

/** Objects of this class represent the summary part of the normal output
 ** from Simplify:  valid, invalid, or unknown.
 ** <p>
 ** 
 ** @see Simplify
 ** @see CECEnum
 ** @see SExp
 **/

public class SimplifyOutputSentinel extends SimplifyOutput {
  int number;

  //@ requires kind == VALID || kind == INVALID || kind == UNKNOWN;
  SimplifyOutputSentinel(int kind, int number) {
    super(kind);
    this.number = number;
  }

  public String toString() {
    return super.toString() + " number=" + number;
  }
}
