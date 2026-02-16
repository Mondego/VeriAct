/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

/**
 * An object of this class represent a progress comment produced by
 * Simplify.
 * 
 * @see Simplify
 * @see escjava.prover.CECEnum
 * @see SExp
 */

public class SimplifyComment extends SimplifyOutput
{
    /*@ spec_public @*/ final /*@ non_null */ String msg;

    //@   ensures this.msg == msg;
    SimplifyComment(/*@ non_null */ String msg) {
        super(COMMENT);
        this.msg = msg;
    }

    //@ ensures \result == msg;
    public /*@ pure @*/ /*@ non_null */ String getMsg() {
        return msg;
    }
}
