/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;

/** Objects of this class represent possible normal outputs from Simplify.
 ** <p>
 ** 
 ** @see Simplify
 ** @see CECEnum
 ** @see SExp
 **/

public class SimplifyOutput {
  public static final int VALID = 0;
  public static final int INVALID = VALID + 1;
  public static final int UNKNOWN = INVALID + 1;

  public static final int COMMENT = UNKNOWN + 1;

  public static final int COUNTEREXAMPLE = COMMENT + 1;
  public static final int EXCEEDED_PROVER_KILL_TIME = COUNTEREXAMPLE + 1;
  public static final int EXCEEDED_PROVER_KILL_ITER = EXCEEDED_PROVER_KILL_TIME + 1;
  public static final int REACHED_CC_LIMIT = EXCEEDED_PROVER_KILL_ITER + 1;
  public static final int EXCEEDED_PROVER_SUBGOAL_KILL_TIME = REACHED_CC_LIMIT + 1;
  public static final int EXCEEDED_PROVER_SUBGOAL_KILL_ITER = EXCEEDED_PROVER_SUBGOAL_KILL_TIME + 1;

  public static final int WARNING_TRIGGERLESS_QUANT = EXCEEDED_PROVER_SUBGOAL_KILL_ITER + 1;

  public static final int END = WARNING_TRIGGERLESS_QUANT + 1;

  int kind;  // one of the above
  //@ invariant VALID <= kind && kind < END;

  //@ ensures VALID <= \result && \result < END;
  public int getKind() {
    return kind;
  }

  SimplifyOutput(int kind) {
    this.kind = kind;
  }

  public String toString() {
    return super.toString() + " kind=" + kind;
  }
}
