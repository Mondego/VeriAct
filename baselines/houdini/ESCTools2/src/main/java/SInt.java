/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

import java.io.*;

/**
 * <code>SInt</code>s are S-expressions representing integers.
 */

/**
 * @design unknown author - The "public" below was added so that stuff
 * in the sexpgrep package can know about SInt (for the conversion
 * between the two kinds of s-expressions).
 */

final /* was package */ public class SInt extends SExp
{
    /**
     * The <code>int</code> we represent.
     */
    private int value;

    /**
     * Create a <code>SInt</code> representing a given
     * <code>int</code>.  Clients are allowed to create
     * <code>SInt</code>s only by calling <code>fromInt</code> so we
     * can choose later to intern if we wish.
     */
    //@ private normal_behavior
    //@   modifies value;
    //@   ensures value == i;
    private SInt(int i) {
	value = i;
    }

    /**
     * Create a <code>SInt</code> representing a given
     * <code>int</code>.
     */
    static /*@ pure non_null @*/ SInt fromInt(int i) {
	return new SInt(i);
    }

    /**
     * @return a flag indicating if we represent an integer, which is
     * always <code>true</code>.
     */
    //@ also
    //@ public normal_behavior
    //@   ensures \result;
    public /*@ pure @*/ boolean isInteger() {
	return true;
    }

    /**
     * If we represent an integer, return it as an <code>int</code>;
     * otherwise, throw SExpTypeError.
     */
    //@ also
    //@ private normal_behavior
    //@   ensures \result == value;
    public /*@ pure @*/ int getInteger() {
	return value;
    }

    /**
     * @return <code>true</code> iff parameter <code>o</code> is an
     * <code>SInt</code> with the same value as <code>this</code>.
     */
    public /*@ pure @*/ boolean equals(Object o) {
        if (!(o instanceof SInt)) return false;
        SInt a = (SInt)o;
        return this.value == a.value;
    }

    /**
     * Print a textual representation of us on a given
     * <code>PrintStream</code>.
     *
     * <p> Note: This routine will take a <code>PrintWriter</code>
     * instead when we switch to a more recent version of JDK. </p>
     */
    public /*@ pure @*/ void print(/*@ non_null @*/ PrintStream out) {
	out.print(value);
    }

    /**
     * Pretty-print a textual representation of us on a given
     * <code>PrintStream</code>.
     *
     * <p> Note: This routine will take a <code>PrintWriter</code>
     * instead when we switch to a more recent version of JDK. </p>
     */
    public /*@ pure @*/ void prettyPrint(/*@ non_null @*/ PrintStream out) {
	out.print(value);
    }
}
