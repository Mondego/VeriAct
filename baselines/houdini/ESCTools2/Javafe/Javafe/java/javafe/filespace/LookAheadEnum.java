/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;
import java.util.NoSuchElementException;


/**
 * This layer describes how to implement an {@link Enumeration} in
 * terms of a single function that returns the next element in a
 * series, or null if the series is exhausted.<p>
 *
 * Using one function instead of the two used by {@link Enumeration}
 * has the advantage of avoiding possible duplication of code to
 * determine whether or not any elements remain in the series.<p>
 *
 * Limitation: <code>null</code> cannot belong to the resulting
 * {@link Enumeration}s.<p>
 */

abstract class LookAheadEnum implements Enumeration {

    //@ invariant !returnsNull;

    // Object invariant: If lookAheadValid is true, then
    // calcNextElement has already been called and its result slashed
    // in lookAhead.
    
    // In particular, if lookAheadValid is true and lookAhead == null,
    // then we are exhausted.  If lookAheadValid is true and lookAhead
    // is non-null, then lookAhead contains the next element of the
    // series.


    //@ invariant lookAheadValid ==> moreElements == (lookAhead != null);
    //@ spec_public
    private boolean lookAheadValid = false;

    //@ invariant \typeof(lookAhead) <: elementType || lookAhead == null;
    //@ spec_public
    private Object lookAhead = null;

    /**
     * Ensure that lookAheadValid is set, calling calcNextElement if
     * needed.
     */
    //@ modifies lookAheadValid, lookAhead;
    //@ ensures lookAheadValid;
    private void ensureLookedAhead() {
	if (!lookAheadValid) {
	    lookAhead = calcNextElement();
	    /*
	     * We assume that moreElements magically predicted
	     * calcNextElement()'s output:
	     */
	    //@ assume moreElements == (lookAhead != null);
	    lookAheadValid = true;
        }
    }
	

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Create an look-ahead enumerator that will return the non-null
     * results of calcNextElement().
     */
    //@ ensures !lookAheadValid;
    //@ ensures lookAhead == null;		// So subclass can set elementType...
    public LookAheadEnum() {
	//@ set elementType = \type(Object);	// override in subclasses...
	//@ set returnsNull = false;
    }

    /**
     * Create a look-ahead enumerator that will return first followed
     * by the non-null results of calcNextElement().
     */
    //@ requires first != null;
    //@ ensures moreElements;
    //   So subclass can set elementType...:
    //@ ensures \typeof(lookAhead) <: \typeof(first);
    public LookAheadEnum(Object first) {
	//@ set elementType = \type(Object);	// override in subclasses...
	//@ set returnsNull = false;

	lookAheadValid = true;
	lookAhead = first;

    }


    /***************************************************
     *                                                 *
     * Handling the Enumeration interface:	       *
     *                                                 *
     **************************************************/

    /**
     * @return true iff any more elements exist in this enumeration.
     */
    //@ also
    //@ private behavior
    //@   modifies lookAheadValid, lookAhead;
    //@   ensures \result == (lookAhead != null);
    public final boolean hasMoreElements() {
	ensureLookedAhead();

	return lookAhead != null;
    }

    //@ also
    //@ public behavior
    //@   modifies \everything;
    	//@   signals_only NoSuchElementException;
    //@   signals (NoSuchElementException) (lookAhead == null);
    //    Should be:
    //    modifies lookAheadValid, lookAhead;
    //    but because NoSuchElementException has no spec, then the
    //    constructor has the default modifies clause of \everything.
    public final Object nextElement() {
	ensureLookedAhead();

	if (lookAhead == null)
	    throw new NoSuchElementException();

	lookAheadValid = false;
	return lookAhead;
    } //@ nowarn Exception;


    /***************************************************
     *                                                 *
     * Calculating the next element:		       *
     *                                                 *
     **************************************************/

    /**
     * Compute the next element in the series, or return null if the
     * series is exhausted.
     *
     * This function will never be called again once it returns null.
     */
    //---@ modifies \nothing;
    //@ ensures \typeof(\result) <: elementType || \result==null;
    protected abstract Object calcNextElement();

}
