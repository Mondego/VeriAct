/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.util.Vector;

/**
 * A stack of Vector objects.
 *
 * <p> Contains 1 or more Vectors arranged in a stack.  Direct access
 * is allowed only to the elements of the top Vector via a subset of
 * the usual java.util.Vector operations.  These Vectors may not
 * contain null elements.<p>
 *
 * <p> push() pushs a zero-length Vector on the stack, pop() discards
 * the top Vector, and merge() combines the top two Vectors.
 *
 * <p> The caller is responsible for ensuring that pop() and merge()
 * are called only when the stack has at least 2 Vectors on it.
 *
 * <p> All the elements in the Vectors must be of type elementType.
 *
 * <p> Note: Although this class as a whole is thread safe, individual
 * instances are not.  Thus, different threads can safely access
 * different instances of <CODE>StackVector</CODE> without any
 * synchronization, but different threads accessing the <EM>same</EM>
 * instance will have to lock and synchronize.
 */

public final class StackVector
{
    /***************************************************
     *                                                 *
     * Private instance fields:			       *
     *                                                 *
     **************************************************/

    /** The type of our elements */
    //@ ghost public \TYPE elementType;

    /**
     * Our data representation is as follows: <p>
     *
     * elements[0], ..., elements[elementCount-1] contain
     * the Vectors on our stack, stored as sequences of elements from
     * the bottom of the stack (pushed longest ago) to the top of the
     * stack.  Each sequence is seperated by a null element.<p>
     *
     * Example: [1, 2, null, 3, 4, null, 5, 6] represents the stack [1,
     * 2], [3, 4], [5, 6] where the Vector [5, 6] is on top so
     * elementAt(1) will return 6. <p>
     *
     * To speed access to the top Vector, currentStackBottom points to
     * the first element of the top Vector (5 in the example).  Note
     * that this may point just beyond the last element of elements if
     * the top Vector is zero-length.<p>
     *
     * vectorCount holds the # of Vectors; it exists so that clients
     * can make sure they've left a StackVector the way they found it
     * and so that preconditions can be written for pop(), etc.<p>
     */

    //@ invariant elements != null;
    //@ invariant elements.length>0;
    //@ invariant \typeof(elements) == \type(Object[]);

    //@ invariant elements.owner == this;
    private /*@ spec_public */ Object[] elements = new Object[10];


    /*@ invariant 0<=elementCount && elementCount <= elements.length
      @		&&  (\forall int i; 0<=i && i<elementCount
      @			==> elements[i]==null || \typeof(elements[i]) <: elementType)
      @ 	&& 0<=currentStackBottom && currentStackBottom<=elementCount
      @		&& (currentStackBottom == 0 ||
      @			elements[currentStackBottom-1]==null); 
      @*/

    /*@spec_public*/ private int elementCount = 0;


    /*@spec_public*/ private int currentStackBottom = 0;


    //@ invariant vectorCount>=1;
    /*
     * The following invariant is used in the form of assumes as needed,
     * but not checked.  (too hard)
     *
     *   (vectorCount>=2) ==> (currentStackBottom>0)
     */
    /*@spec_public*/ private int vectorCount = 1;


    /***************************************************
     *                                                 *
     * Constructors:				       *
     *                                                 *
     **************************************************/

    /**
     * Create a StackVector that contains only 1 zero-length Vector.
     */
    //@ ensures elementCount == 0;
    //@ ensures vectorCount == 1;
    public StackVector() {
	//@ set elements.owner = this;
    }


    /***************************************************
     *                                                 *
     * Accessing the top Vector:		       *
     *                                                 *
     **************************************************/

    /**
     * Return the size of the top Vector. <p>
     *
     * Indices into the top Vector range from 0 to size()-1.<p>
     *
     */
    //@ ensures \result >= 0 ;
    //@ ensures \result == (elementCount - currentStackBottom);
    //@ pure
    public final int size() {
	return elementCount - currentStackBottom;
    }


    /*
     * Precondition is for the default case where caller does not want
     * ArrayIndexOutOfBoundsException raised.
     *
     * If caller does want to deal with this, use nowarn Pre on calls to
     * these functions.
     */
    //@ requires 0<=index && index+currentStackBottom<elementCount;
    private void checkBounds(int index)
		/*throws ArrayIndexOutOfBoundsException*/ {
	if (index < 0 || index+currentStackBottom >= elementCount) {
	    throw new ArrayIndexOutOfBoundsException(index);
	}
    }

    /**
     * Return the element in the top Vector at the given index. <p>
     *
     * @exception ArrayIndexOutOfBoundsException if the index is out of
     * range.
     */
    //@ requires 0<=i && i+currentStackBottom<elementCount;
    //@ ensures \typeof(\result) <: elementType;
    //@ ensures \result != null;
    public Object elementAt(int i) 
		/*throws ArrayIndexOutOfBoundsException*/ {
	checkBounds(i);
	return elements[currentStackBottom + i];
    }		//@ nowarn Post;		// (thinks could be null)

    //@ requires 0<=i && i+currentStackBottom<elementCount;
    //@ requires o == null ||  (\typeof(o) <: elementType);
    public void setElementAt(Object o, int i)
		/*throws ArrayIndexOutOfBoundsException*/ {
	checkBounds(i);
	elements[currentStackBottom + i] = o;
    }	

    /**
     * Add x to the end of the top Vector. <p>
     *
     * x may be null, in which case the caller needs to cleanup to
     * ensure that a null element does not stay in the top Vector. <p>
     */
    //@ requires x==null || \typeof(x) <: elementType;
    //@ modifies elementCount, elements, elements[elementCount];
    //@ ensures elementCount == \old(elementCount)+1;
    //@ ensures elementCount>0 && elements[elementCount-1]==x;
    private void addElementInternal(Object x) {
	if (elementCount >= elements.length) {
	    Object[] newElements = new Object[2*elementCount];
	    //@ set newElements.owner = this;

	    System.arraycopy(elements, 0, newElements, 0, elements.length);
	    elements = newElements;
	}

	elements[elementCount++] = x;
    }


    /**
     * Add an element at the end of the top Vector. <p>
     *
     */
    //@ requires x != null;
    //@ requires \typeof(x) <: elementType;
    //@ modifies elementCount, elements, elements[*];
    /*@ ensures (elementCount - currentStackBottom) ==
	        (\old(elementCount) - currentStackBottom) + 1; */
    public final void addElement(/*@ non_null */ Object x) {
	Assert.precondition(x != null);
	addElementInternal(x);
    }


  /*
    public final void addElements(Enumeration e) {
    while( e.hasMoreElements() )
      addElement( e.nextElement() );
  }
  */


    /** Zero the top Vector. */
    //@ modifies elementCount, elements[*];
    //@ ensures elementCount == currentStackBottom;
    public final void removeAllElements() {
	for (int i = currentStackBottom; i < elementCount; i++)
	    elements[i] = null; // Allow for GC
	elementCount = currentStackBottom;
    }


    //@ requires dst != null;
    //@ requires (elementCount - currentStackBottom) <= dst.length;
    //@ modifies dst[*];
    /*@ ensures (\forall int i; 0 <= i && i < elementCount-currentStackBottom
			==> dst[i] != null); */
    public final void copyInto(Object[] dst) {
	System.arraycopy(elements, currentStackBottom,
			 dst, 0,
			 elementCount - currentStackBottom);
    }	//@ nowarn Post;


    /**
     * Return true iff the top Vector contains o.
     */
    public final boolean contains(Object o) {
	for( int i=currentStackBottom; i<elementCount; i++ ) {
	    if( elements[i] == o ) return true;
	}
	return false;
    }


    /***************************************************
     *                                                 *
     * Stack manipulation methods:		       *
     *                                                 *
     **************************************************/

    /**
     * Reset us to the state where we contain only 1 Vector, which has
     * zero-length.
     */
    //@ modifies vectorCount;
    //@ ensures vectorCount == 1;
    //@ modifies elementCount, currentStackBottom;
    //@ ensures elementCount == 0;
    //@ ensures currentStackBottom == 0;
    public void clear() {
	elementCount = currentStackBottom = 0;
	vectorCount = 1;
    }


    /**
     * Return the number of Vectors on our stack. <p>
     *
     */
    //@ ensures \result==vectorCount; 
    public final int vectors() {
	return vectorCount;
    }


    /**
     * Push a zero-length Vector.
     *
     */
    //@ modifies vectorCount, currentStackBottom, elementCount, elements, elements[*];
    //@ ensures vectorCount == \old(vectorCount)+1;
    //@ ensures currentStackBottom == elementCount;
    public void push() {
	addElementInternal(null);
	currentStackBottom = elementCount;
	vectorCount++;
    }

    /**
     * Pop off the current top Vector. <p>
     *
     * Precondition: at least 2 Vectors are on our stack.<p>
     */
    //@ requires vectorCount>=2;
    //@ modifies vectorCount;
    //@ ensures vectorCount == \old(vectorCount)-1;
    //@ modifies elementCount, currentStackBottom;
    public void pop() {
	//@ assume (vectorCount>=2) ==> (currentStackBottom>0); // "invariant"

	Assert.precondition(currentStackBottom>0);

	int i = currentStackBottom - 1;
	//@ assert elements[i]==null;

	elementCount = i;
	for( ; i > 0; i--)
	    if (elements[i-1] == null)
		break;
	currentStackBottom = i;

	vectorCount--;
    }


    /**
     * Merge the top Vector with the Vector just under it by appending
     * the former to the latter. <p>
     *
     *   Example: ..., A, TOP -> ..., A^TOP <p>
     *
     * Precondition: there are at least two vectors on our stack.<p>
     */
    //@ requires vectorCount>=2;
    //@ modifies vectorCount, elementCount, elements[*];
    //@ ensures vectorCount == \old(vectorCount)-1;
    //@ modifies currentStackBottom;
    public void merge() {
	//@ assume (vectorCount>=2) ==> (currentStackBottom>0); // "invariant"

	Assert.precondition(currentStackBottom>0);
	
	int i = currentStackBottom - 1;
	//@ assert elements[i]==null;

	System.arraycopy(elements, i+1, elements, i, size());
	for( ; i > 0; i--)
	    if (elements[i - 1] == null)
		break;
	currentStackBottom = i;
	elementCount--;

	vectorCount--;
    }

    /**
     ** Returns the contents of all the vectors on the stack as a single
     ** vector.<p>
     **/
    public Vector stackContents() {
        Vector vec = new Vector();
        for (int i = 0; i < elementCount; i++) {
            Object o = elements[i];
            if (o != null) {
                vec.addElement(o);
            }
        }
        return vec;
    }

    public /*@non_null*/String toString() {
	StringBuffer sb = new StringBuffer();
	sb.append("StackVector[");
	for (int i=0; i<elementCount; ++i) {
		Object o = elements[i];
		if (o==null) sb.append(" :: ");
		else {
			sb.append(o.toString());
			sb.append(" ");
		}
	}
	sb.append("]");
	return sb.toString();
    }

}
