/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;

/** An object of this class represent a "result" produced by Simplify.
 ** <p>
 ** 
 ** @see Simplify
 ** @see CECEnum
 ** @see SExp
 **/

public class SimplifyResult extends SimplifyOutput {
  SList labels;
  SList context;

  public SList getLabels() {
    return labels;
  }

  public SList getContext() {
    return context;
  }
  
  //@ requires COUNTEREXAMPLE <= kind && kind < END;
  SimplifyResult(int kind, SList labels, SList context) {
    super(kind);
    this.labels = labels;
    this.context = context;
  }

  public String toString() {
    return super.toString() + " labels=" + labels + " context=" + context;
  }
}
