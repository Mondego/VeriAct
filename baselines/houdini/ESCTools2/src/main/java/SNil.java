/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

import java.io.*;

/**
 * The single <code>SNil</code> instance represents the empty list of
 * <code>SExp</code>s.
 */

final class SNil extends SList
{
    /**
     * The single instance of this class, or <code>null</code> if it
     * has not yet been allocated.
     */
    //@ spec_public
    private static /*@ nullable */ SNil single = null;

    /**
     * Instance creation is private so we can ensure that at most one
     * instance of this class is ever created.
     */
    //@ private normal_behavior
    //@   requires single == null;
    //@   modifies single;
    //@   ensures single != null;
    private SNil() {
	javafe.util.Assert.notFalse(single == null);
	single = this;
    }

    /**
     * @return the single SNil instance.
     */
    //@ public normal_behavior
    //@   modifies single;
    //@   ensures \result != null;
    //@ also
    //@ private normal_behavior
    //@   requires single == null;
    //@   modifies single;
    //@   ensures single != null;
    //@ also
    //@ private normal_behavior
    //@   requires single != null;
    //@   modifies \nothing;
    //@   ensures \result == single;
    public static /*@ non_null @*/ SNil getNil() {
	if (single != null) {
	    return single;
	} else {
	    single = new SNil();
	    return single;
	}
    }

    /**
     * @return true if the parameter is also the single instance of
     * SNil.
     */
    //@ also
    //@ private normal_behavior
    //@   ensures \result <==> (o == this);
    public /*@ pure @*/ boolean equals(/*@ nullable */ Object o) {
        return o == this;
    }

    /**
     * @return a flag indicating if we are we an empty list, which is
     * always <code>true</code> since we are the nil instance.
     */
    //@ also
    //@ public normal_behavior
    //@   ensures \result;
    public /*@ pure @*/ boolean isEmpty() {
	return true;
    }

    /**
     * @return our length, which is zero since we are the nil
     * instance.
     */
    //@ also
    //@ public normal_behavior
    //@   ensures \result == 0;
    public /*@ pure @*/ int length() { return 0; }
}
