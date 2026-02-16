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


/**
 * We represent [C.]this.
 *
 * <p> classPrefix holds C if present, and null otherwise.
 *
 * @note classPrefix will hold a <em>**TypeSig**</em> instead of a
 * TypeName if we have been inferred.  (The inferred field specifies
 * whether or not we have been inferred.)
 */

public class ThisExpr extends Expr
{
  public /*@nullable*/ Type classPrefix;


  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  //* Were we inferred by the disambiguation code?
  public boolean inferred = false;

  /*@ public represents startLoc <- (classPrefix != null && classPrefix instanceof TypeName)
    @ 		? classPrefix.getStartLoc() : loc;
    @*/
    public /*@ pure @*/ int getStartLoc() {
        if (classPrefix != null && classPrefix instanceof TypeName)
	    return classPrefix.getStartLoc();
	return loc;
    }

  //@ also public normal_behavior
  //@ ensures \result == Location.inc(loc, 3);
  public /*@ pure @*/ int getEndLoc() { return Location.inc(loc, 3); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.classPrefix == classPrefix;
  //@ ensures this.loc == loc;
  protected ThisExpr(/*@nullable*/Type classPrefix, int loc) {
     super();
     this.classPrefix = classPrefix;
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

     if (index == 0) return this.classPrefix;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ThisExpr"
        + " classPrefix = " + this.classPrefix
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.THISEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitThisExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitThisExpr(this, o); }

  public void check() {
     super.check();
     if (this.classPrefix != null)
        this.classPrefix.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ThisExpr make(/*@nullable*/Type classPrefix, int loc) {
     ThisExpr result = new ThisExpr(classPrefix, loc);
     return result;
  }
}
