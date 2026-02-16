// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package javafe.ast;

import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.ErrorSet;
import javafe.tc.TagConstants;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit


public class SimpleName extends Name
{
  public /*@non_null*/ Identifier id;


  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  //@ invariant length == 1;

  /**
   * @return a String representation of <code>this</code> in Java's
   * syntax.
   */
  public /*@non_null*/ String printName() {
    return id.toString();
  }

    public boolean equals(/*@ nullable \readonly @*/ Object other) {
	if (other instanceof SimpleName)
	    return id == ((SimpleName)other).id;
	else
	    return false;
    }

  public /*@non_null*/ Name prefix(int len) {
    Assert.precondition(len == 1);
    return make(id, loc);
  }

  /** Return a hash code for <code>this</code> such that two
    <code>Name</code>s that are <code>equals</code> have the same hash
    code.
   */
  public int hashCode() {
    return id.hashCode();
  }

  /**
   * @return the first <code>len</code> identifiers in
   * <code>this</code> in an array.  Requires that <code>len</code> be
   * between 0 and length of <code>this</code> inclusive.
   */
  public String[] toStrings(int len) {
    Assert.precondition(len == 0 || len == 1);
    if (len == 0) return emptyStringArray;
    String[] result = { id.toString() };
    return result;
  }

  /** Return the number of identifiers in <code>this</code>. */
  public int size() { return 1; }

  /** Return the ith identifier of <code>this</code>. */
  public /*@non_null*/ Identifier identifierAt(int i) {
    if (i != 0) throw new ArrayIndexOutOfBoundsException();
    return id;
  }

  /** Return the location of the ith identifier of <code>this</code>. */
  public int locIdAt(int i) {
    if (i != 0) throw new ArrayIndexOutOfBoundsException();
    return loc;
  }

  /** Return the location of the dot after the ith identifier of
    <code>this</code>.
   */
  public int locDotAfter(int i) {
    throw new ArrayIndexOutOfBoundsException();
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.id == id;
  //@ ensures this.loc == loc;
  protected SimpleName(/*@non_null*/ Identifier id, int loc) {
     super();
     this.id = id;
     this.loc = loc;
  }

// Generated boilerplate methods:

  public final int childCount() {
     return 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.id;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[SimpleName"
        + " id = " + this.id
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.SIMPLENAME;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitSimpleName(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitSimpleName(this, o); }

  public void check() {
     super.check();
     if (this.id == null) throw new RuntimeException();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ SimpleName make(/*@non_null*/ Identifier id, int loc) {
     SimpleName result = new SimpleName(id, loc);
     return result;
  }
}
