/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.util.*;

/**
 * A simple implementation of imperative sets.  Set's may not contain
 * null.
 */

public class Set implements Cloneable
{
    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     **************************************************/

    /** Our element type */
    //@ ghost public \TYPE elementType;


    /**
     * We contain the element e iff ht has the mapping e -> e. <p>
     *
     * All mappings of ht are of the form e' -> e' for some e'.
     */
    //+@ invariant ht.keyType == elementType;
    //+@ invariant ht.elementType == elementType;
    //@ invariant ht.owner == this;
    //@ spec_public
    private /*@ non_null @*/ Hashtable ht;


    /***************************************************
     *                                                 *
     * Construction:				       *
     *                                                 *
     **************************************************/

    /**
     * Create an empty set
     */
    public Set() {
	int initCapacity = 5;
	ht = new Hashtable(initCapacity);
	//+@ set ht.keyType = elementType;
	//+@ set ht.elementType = elementType;
	//@ set ht.owner = this;
    }


    /**
     * Create a set from the elements of an Enumeration
     */
    //@ requires e != null;
    //@ requires !e.returnsNull;
    //@ ensures elementType == e.elementType;
    public Set(Enumeration e) {
	this();
	//@ set elementType = e.elementType;
	//+@ set ht.keyType = e.elementType;
	//+@ set ht.elementType = e.elementType;

	while( e.hasMoreElements() ) {
	    Object item = e.nextElement();
	    Assert.notNull(item);
	    add( item );
	}
    }


    /***************************************************
     *                                                 *
     * Inspectors:				       *
     *                                                 *
     **************************************************/

    /**
     * Return the number of elements we hold.
     */
    public int size() {
	return ht.size();
    }

    /**
     * Do we contain no elements?
     */
    public boolean isEmpty() {
	return ht.isEmpty();
    }


    /**
     * Do we contain a particular element?
     */
    //@ requires o != null;
    //@ requires \typeof(o) <: elementType;
    public boolean contains(Object o) {
	return ht.containsKey( o );
    }


    /**
     * Return an enumeration of our elements
     */
    //@ ensures \result != null;
    //@ ensures !\result.returnsNull;
    //@ ensures \result.elementType == elementType;
    public Enumeration elements() {
	return ht.keys();
    }


    public /*@non_null*/String toString() {
	return "[SET: "+ht.toString()+"]";
    }


    /***************************************************
     *                                                 *
     * Manipulators:				       *
     *                                                 *
     **************************************************/

    /**
     * Remove all our elements
     */
    public void clear() {
	ht.clear();
    }


    /**
     * Add an element
     *
     * Return 'true' iff the element was already in the set.
     */
    //@ requires o != null;
    //@ requires \typeof(o) <: elementType;
    public boolean add(Object o) {
	return ht.put( o, o ) != null;
    }

    /**
     * Add all the elements of a given enumeration
     */
    //@ requires e != null;
    //@ requires !e.returnsNull;
    //@ requires e.elementType <: elementType;
    public void addEnumeration(Enumeration e) {
	while( e.hasMoreElements() ) {
	    Object item = e.nextElement();
	    Assert.notNull(item);
	    add( item );
	}
    }


    /**
     * Remove a particular element if we contain it.  Return true
     * iff the element was previously in the set.
     */
    //@ requires o != null;
    //@ requires \typeof(o) <: elementType;
    public boolean remove(Object o) {
	return ht.remove(o) != null;
    }


    /**
     * Remove all elements not contained in another set.  This has the
     * effect of setting us to the intersection of our original value
     * and the other set.
     */
    //@ requires s != null;
    //@ requires s.elementType == elementType;
    public void intersect(Set s) {
	Enumeration e = ht.keys();
	while( e.hasMoreElements() ) {
	    Object o = e.nextElement();
	    if (!s.ht.containsKey(o))
		ht.remove(o);
	}
    }


    /**
     * Adds all elements  in another set.  This has the
     * effect of setting us to the union of our original value
     * and the other set. Return true if any values were added.
     */
    //@ requires s != null;
    //@ requires s.elementType == elementType;
    public boolean union(Set s) {
	boolean changed = false;
	for(Enumeration e = s.elements(); e.hasMoreElements(); ) {
	    Object o = e.nextElement();
	    if (!ht.containsKey(o)) {
		changed = true;
		ht.put(o,o);
	    }
	}
	return changed;
    }

    /**
     * Returns whether or not the set has any element in common with
     * <code>s</code>.  Thus, if the intersection of the sets is empty,
     * that is, if the sets are disjoint, then <code>false</code> is returned.
     * The operation leaves both sets unchanged.
     */
    //@ requires s.elementType == elementType;
    public boolean containsAny(/*@non_null*/ Set s) {
      Enumeration e;
      if (size() < s.size()) {
	e = ht.keys();
      } else {
	e = s.ht.keys();
	s = this;
      }
      while (e.hasMoreElements()) {
	Object o = e.nextElement();
	if (s.ht.containsKey(o)) {
	  return true;
	}
      }
      return false;
    }

    public /*@non_null*/ Object clone() {
      try {
	Set s = (Set)super.clone();
	//@ assume s.elementType == this.elementType;
	Enumeration e = this.elements();
	s.addEnumeration(e);
	return s;
      } catch (CloneNotSupportedException c) {
	// shouldn't happen, since Set implements Cloneable
	throw new IllegalStateException("unreachable code reached!");
      }
    }

}
