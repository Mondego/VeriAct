/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


import java.io.*;


/**
 ** The single <code>SNil</code> instance represents the empty list of
 ** <code>SExp</code>s. <p>
 **/

final class SNil extends SList {

    /***************************************************
     *                                                 *
     * Class Variables:				       *
     *                                                 *
     ***************************************************/

    /**
     ** The single instance of this class or null if it has not yet been
     ** allocated.
     **/
    private static SNil single = null;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Instance creation is private so we can ensure that at most one
     ** instance of this class is ever created.
     **/
    //@ requires single==null
    private SNil() {
	javafe.util.Assert.notFalse(single==null);
	single = this;
    }

    /**
     ** Return the single SNil instance.
     **/
    //@ ensures RES!=null
    public static SNil getNil() {
	if (single!=null) {
	    return single;
	} else {
	    single = new SNil();
	    return single;
	}
    }
    

    /**
     ** Return true if 0 is also the single instance of SNil.
     **/
    public boolean equals(Object o) {
        return o == this;
    }
    


    /***************************************************
     *                                                 *
     * List Accessors:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Are we an empty list?
     **/
    public boolean isEmpty() {
	return true;
    }

    /**
     ** Return our length
     **/
    public int length() { return 0; }
}
