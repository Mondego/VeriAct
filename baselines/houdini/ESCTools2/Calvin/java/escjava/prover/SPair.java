/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


import java.io.*;


/**
 ** A <code>SPair</code> is a pair of a <code>SExp</code> and a
 ** <code>SList</code>; together with the <code>SNil</code> class, it is
 ** used to implement lists of <code>SExp</code>s. <p>
 **
 ** Considered as <code>SList</code>s, <code>SPair</code>s always
 ** represent non-empty lists.<p>
 **/

/*package*/ class SPair extends SList {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     ***************************************************/

    /**
     ** The head of our list; this field should not be modified by
     ** clients and should be non-null.
     **/
    //@ invariant head!=null
    public SExp head;

    /**
     ** The tail of our list; this field should not be modified by
     ** clients and should be non-null.
     **/
    //@ invariant tail!=null
    public SList tail;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     ***************************************************/

    /**
     ** Return a new list with given head and tail. <p>
     **
     ** Both must be non-null.<p>
     **/
    //@ requires head!=null && tail!=null
    public SPair(SExp head, SList tail) {
	this.head = head;
	this.tail = tail;
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
	return false;
    }

    /**
     ** If we represent a non-empty list, return it as a
     ** <code>SPair</code>; otherwise, throw <code>SExpTypeError</code>.
     **/
    /*package*/ SPair getPair() {
	return this;
    }

    /**
     ** Return our length
     **/
    public int length() {
	return 1+tail.length();
    }
    
    /**
     ** Return true if the heads are equal and the tails are equal.
     **/
    public boolean equals(Object o) {
        if (!(o instanceof SPair)) return false;
        SPair a = (SPair)o;
        return this.head.equals(a.head) && this.tail.equals(a.tail);
    }
        
}
