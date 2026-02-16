/* Copyright Hewlett-Packard, 2002 */

/*
 * File: HavocAssumeVec.java <p>
 *
 * This file is a template for a vector of X.  It is filled in using
 * sed (e.g, see the Makefile for javafe.ast).  The following
 * replacements need to be done:<p>
 *
 * HavocAssume should be globally replaced with the simple name of the
 *		element type<p>
 * escjava.translate should be globally replaced with the package the resulting
 *		Vector type should live in<p>
 * escjava.translate should be globally replaced with the package the
 *		element type belongs to<p>
 *
 * The resulting type will have name HavocAssumeVec.<p>
 * 
 * Example: to make a vector of java.lang.Integer's that lives in
 * javafe.ast, substitute Integer, javafe.ast, and java.lang
 * respectively.<p>
 *
 *
 * By Default, HavocAssumeVec's may not contain nulls.  If the ability to
 * hold nulls is desired, remove all lines with "// @@@@" on them.<p>
 */




/**
 ** HavocAssumeVec: A Vector of HavocAssume's. <p>
 **
 ** The interface to this class follows java.util.Vector, where
 ** appropriate. <p>
 **
 **
 ** This class was automatically created from the template
 ** javafe.util/_<tt>TYPE</tt>_Vec.j.<p>
 **
 ** DO NOT EDIT!!<p>
 **/

package escjava.translate;


import java.util.Vector;
import javafe.util.StackVector;

import escjava.translate.HavocAssume;


public class  HavocAssumeVec {

    /***************************************************
     *                                                 *
     * Instance fields:				       *
     *                                                 *
     ***************************************************/

    private /*@ non_null @*/ HavocAssume[] elements;
    //@ invariant (\forall int i; (0<=i && i<count) ==> elements[i]!=null) // @@@@
    //@ invariant \typeof(elements) == \type(HavocAssume[])

    //@ invariant elements.owner == this

    /*@spec_public*/ private int count;
    //@ invariant 0 <= count;
    //@ invariant count <= elements.length;


    /***************************************************
     *                                                 *
     * Private constructors:			       *
     *                                                 *
     ***************************************************/

    //@ requires els!=null
    //@ requires \nonnullelements(els)				// @@@@
    //@ ensures count == els.length
    private HavocAssumeVec(HavocAssume[] els) {
	this.count = els.length;
	this.elements = new HavocAssume[count];
	//@ set elements.owner = this

	System.arraycopy(els,0, elements,0, count);
    }

    //@ requires cnt >= 0
    //@ ensures this.elements.length >= cnt
    private HavocAssumeVec(int cnt) {
	this.elements = new HavocAssume[(cnt == 0 ? 2 : cnt)];
	//@ set elements.owner = this

	this.count = 0;
    }


    /***************************************************
     *                                                 *
     * Public maker methods:			       *
     *                                                 *
     ***************************************************/

    //@ ensures \result!=null
    public static HavocAssumeVec make() { 
	return new HavocAssumeVec(0);
    }

    //@ requires count >= 0
    //@ ensures \result!=null
    public static HavocAssumeVec make(int count) { 
	return new HavocAssumeVec(count);
    }

    //@ requires vec!=null
    //@ requires vec.elementType <: \type(HavocAssume)
    //@ requires !vec.containsNull
    //@ ensures \result!=null
    public static HavocAssumeVec make(Vector vec) {
	int sz = vec.size();
	HavocAssumeVec result = new HavocAssumeVec(sz);
	result.count = sz;
	vec.copyInto(result.elements);
	return result;
    }

    //@ requires els!=null
    //@ requires \nonnullelements(els)			// @@@@
    //@ ensures \result!=null
    //@ ensures \result.count == els.length
    public static HavocAssumeVec make(HavocAssume[] els) {
	return new HavocAssumeVec(els);
    }

    // These are from pop() on s:
    //@ modifies s.vectorCount
    //@ ensures s.vectorCount == \old(s.vectorCount)-1
    //@ modifies s.elementCount, s.currentStackBottom
    //
    //@ requires s.vectorCount>1
    //@ requires s.elementType <: \type(HavocAssume)
    //@ ensures \result!=null
    //@ ensures \result.count == (\old(s.elementCount) - \old(s.currentStackBottom))
    public static HavocAssumeVec popFromStackVector(/*@ non_null @*/ StackVector s) {
	// Creates a new HavocAssumeVec from top stuff in StackVector
	int sz = s.size();
	HavocAssumeVec r = new HavocAssumeVec(sz);
	s.copyInto(r.elements);
	r.count = sz;
	s.pop();
	return r;
    }


    /***************************************************
     *                                                 *
     * Other methods:				       *
     *                                                 *
     ***************************************************/

    //@ requires 0<=index && index<count
    //@ ensures \result!=null					// @@@@
    public final HavocAssume elementAt(int index)
	  /*throws ArrayIndexOutOfBoundsException*/ {
	if (index < 0 || index >= count)
	    throw new ArrayIndexOutOfBoundsException(index);

	javafe.util.Assert.notNull(elements[index]);		// @@@@
	return elements[index];
    }

    //@ requires 0<=index && index<count
    //@ requires x!=null					// @@@@
    public final void setElementAt(HavocAssume x, int  index) 
	/*throws ArrayIndexOutOfBoundsException*/ {
	if (index < 0 || index >= count)
	    throw new ArrayIndexOutOfBoundsException(index);
	elements[index] = x;
    }

    //@ ensures \result==count
    public final int size() { return count; }

    public final String toString() {
	StringBuffer b = new StringBuffer();
	b.append("{HavocAssumeVec");
	for(int i = 0; i < count; i++) {
	    b.append(" ");
	    if (elements[i]==null)
		b.append("null");
	    else
		b.append(elements[i].toString());
	}
	b.append('}');
	return b.toString();
    }

  //@ ensures \result!=null
  //@ ensures \nonnullelements(\result)					// @@@@
  public final HavocAssume[] toArray() {
      int ct = count;
      HavocAssume[] b = new HavocAssume[ ct ];
      for(int i = 0; i < ct; i++) {
	  b[i]=elements[i];
      }
      return b;
  }

  //@ ensures \result!=null
  public final HavocAssumeVec copy() {
    HavocAssumeVec result = new HavocAssumeVec(count);
    result.count = count;
    System.arraycopy(elements,0,result.elements,0,count);
    return result;
  }

  //@ requires x!=null						// @@@@
  public boolean contains(HavocAssume x) {
    for(int i = 0; i < count; i++) {
      if( elements[i] == x ) return true;
    }
    return false;
  }

  //@ requires x!=null						// @@@@
  //@ modifies count
  //@ ensures count==\old(count)+1
  public final void addElement(HavocAssume x) {
    if( count == elements.length ) {
      HavocAssume[] newElements = new HavocAssume[ 2*(elements.length==0 ?
					      2 : elements.length) ];
      //@ set newElements.owner = this

      System.arraycopy(elements, 0, newElements, 0, elements.length );
      elements = newElements;
    }
    elements[count++]=x;
  }

  //@ requires x!=null						// @@@@
  //@ modifies count
  public final boolean removeElement(HavocAssume x) {
    for( int i=0; i<count; i++ ) {
      if( elements[i] == x ) {
	  int ct=count;
	for( int j=i+1; j<ct; j++ ) 
	  elements[j-1]=elements[j];
	count--;
	return true;
      }
    }
    return false;
  }

    //@ requires 0<=i && i<count
    //@ modifies count
    public final void removeElementAt(int i) {
	int ct=count;
        for (int j=i+1; j<ct; j++) {
            elements[j-1]=elements[j];
	}	    
        count--;
    }

    //@ requires count>0
    //@ modifies count
    //@ ensures count==\old(count)-1
    //@ ensures \result!=null					// @@@@
    public final HavocAssume pop() {
        count--;
        HavocAssume result = elements[count];
	elements[count] = null;
	return result;
    }


    //@ modifies count
    //@ ensures count==0
    public final void removeAllElements() {
	count = 0;
    }


  //@ requires 0 <= index && index <= this.count
  //@ requires obj!=null					// @@@@
  //@ modifies count
  //@ ensures count==\old(count)+1
  public final void insertElementAt(HavocAssume obj, int index) {
    if( count == elements.length ) {
      HavocAssume[] newElements = new HavocAssume[ 2*(elements.length==0 ?
					      2 : elements.length) ];
      //@ set newElements.owner = this					      
      System.arraycopy(elements, 0, newElements, 0, elements.length );
      elements = newElements;
    }
    int ct=count;
    //@ loop_predicate i!=ct ==> elements[ct] != null, i<= ct
    for( int i=count; i>index; i--) 
      elements[i]=elements[i-1];
    elements[index]=obj;
    count++;
  }

  //@ requires vec!=null
  //@ modifies count
  //@ ensures count==\old(count)+vec.count
  public final void append(HavocAssumeVec vec) {
      //@ loop_predicate count == \old(count)+i, i <= vec.count
    for( int i=0; i<vec.size(); i++)
      addElement( vec.elementAt(i) );
  }
}
