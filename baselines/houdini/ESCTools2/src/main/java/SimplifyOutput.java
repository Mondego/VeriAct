/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

/**
 * Objects of this class represent possible normal outputs from Simplify.
 * 
 * @see Simplify
 * @see escjava.prover.CECEnum
 * @see SExp
 */

public class SimplifyOutput
{
    // Constants representing output types.
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

    // What kind of output are we?
    /*@ spec_public @*/ int kind;
    //@ invariant VALID <= kind && kind < END;

    //@ normal_behavior
    //@   ensures VALID <= \result && \result < END;
    public /*@ pure @*/ int getKind() {
        return kind;
    }

    //@ normal_behavior
    //@   requires VALID <= kind && kind < END;
    //@   modifies this.kind;
    //@   ensures this.kind == kind;
    public SimplifyOutput(int kind) {
        this.kind = kind;
    }

    public /*@ non_null @*/ String toString() {
        return super.toString() + " kind=" + kind;
    }
}
