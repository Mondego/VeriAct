/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;

import java.util.*;

/**
 * This class simply implements an enumeration with no elements.
 */

class EmptyEnum implements Enumeration
{
    //@ represents moreElements <- false;

    //@ invariant !moreElements;
    //@ invariant !returnsNull;

    EmptyEnum() {
	//@ set returnsNull = false;
	//@ set elementType = \type(Object);
    }

    /**
     * @return true iff any more elements exist in this enumeration.
     */
    //@ also
    //@ public normal_behavior
    //@   ensures_redundantly \result == false;
    //@ pure
    public boolean hasMoreElements() {
	return false;
    }

    /**
     * Always throws a NoSuchElementException since there are no
     * elements.
     */
    //@ also
    //@ public exceptional_behavior
    //@   signals_only NoSuchElementException;
    //@   signals (NoSuchElementException) true;
    //@ pure
    public Object nextElement() {
      throw new NoSuchElementException();
    } //@ nowarn Exception;
}
