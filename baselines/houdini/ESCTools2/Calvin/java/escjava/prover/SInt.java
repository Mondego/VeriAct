/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


import java.io.*;


/**
 ** <code>SInt</code>s are S-expressions representing integers. <p>
 **/

/** NOTE:  I added the "public" below so that stuff in the sexpgrep
    package can know about SInt (for the conversion between the two kinds
    of s-expressions).  **/

final /*package*/ public class SInt extends SExp {

    /***************************************************
     *                                                 *
     * Instance methods:			       *
     *                                                 *
     ***************************************************/

    //* the <code>int</code> we represent.
    private int value;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Create a <code>SInt</code> representing a given
     ** <code>int</code>.  Clients are allowed to create
     ** <code>SInt</code>s only by calling <code>fromInt</code> so we can
     ** choose later to intern if we wish.
     **/
    private SInt(int i) {
	value = i;
    }

    /**
     ** Create a <code>SInt</code> representing a given
     ** <code>int</code>.
     **/
    //@ ensures RES!=null
    static SInt fromInt(int i) {
	return new SInt(i);
    }


    /***************************************************
     *                                                 *
     * Simple Accessors:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Do we represent an integer?
     **/
    public boolean isInteger() {
	return true;
    }


    /**
     ** If we represent an integer, return it as an <code>int</code>;
     ** otherwise, throw SExpTypeError.
     **/
    public int getInteger() {
	return value;
    }


    /**
     ** Return true iff o is an SInt with the same value as this.
     **/
    public boolean equals(Object o) {
        if (!(o instanceof SInt)) return false;
        SInt a = (SInt)o;
        return this.value == a.value;
    }


    /***************************************************
     *                                                 *
     * Printing:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Print a textual representation of us on a given
     ** <code>PrintStream</code>. <p>
     **
     ** Note: This routine will take a <code>PrintWriter</code> instead
     ** when we switch to a more recent version of JDK.<p>
     **/
    public void print(PrintStream out) {
	out.print(value);
    }

    /**
     ** Pretty print a textual representation of us on a given
     ** <code>PrintStream</code>. <p>
     **
     ** Note: This routine will take a <code>PrintWriter</code> instead
     ** when we switch to a more recent version of JDK.<p>
     **/
    public void prettyPrint(PrintStream out) {
	out.print(value);
    }
}
