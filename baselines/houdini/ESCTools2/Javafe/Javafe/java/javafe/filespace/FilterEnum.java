/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;

/**
 * A FilterEnum filters an underlying Enumeration using a client
 * supplied Filter.
 */


class FilterEnum extends LookAheadEnum {

    /** The underlying Enumeration: */
    //@ invariant underlyingEnum != null;
    //@ invariant !underlyingEnum.returnsNull;
    //@ invariant underlyingEnum.owner == this;
    protected Enumeration underlyingEnum;

    // We inherit our properties from the underlying Enumeration:
    //@ invariant elementType == underlyingEnum.elementType;


    /** The filter we are using: */
    //@ invariant filter != null;
    //@ invariant filter.acceptedType == elementType;
    public Filter filter;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Filter the Enumeration E using Filter F:
     */
    //@ requires E != null && F != null;
    //@ requires !E.returnsNull;
    //@ requires E.owner == null;
    //@ requires E.elementType == F.acceptedType;
    //@ ensures elementType == E.elementType;
    public FilterEnum(Enumeration E, Filter F) {
	super();

	//@ set E.owner = this;
	underlyingEnum = E;
	filter = F;

	//@ set elementType = E.elementType;
    }


    /***************************************************
     *                                                 *
     * Calculating the next element:		       *
     *                                                 *
     **************************************************/

    /**
     * Compute the next element in the series, or return null if the
     * series is exhausted.
     */

    protected Object calcNextElement() {
	while (underlyingEnum.hasMoreElements()) {
	    Object next = underlyingEnum.nextElement();

	    if (filter.accept(next))
		return next;
        }

	return null;
    }
}
