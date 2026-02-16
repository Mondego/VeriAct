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
 * Represents a ObjectDesignator of the form "TypeName . " 
 *
 * <p> Is created from AmbiguousVariableAccess/AmbiguousMethodInvocation
 * by the name resolution code.  
 * The <code>type</code> must be an instance of either <code>TypeName</code>
 * or <code>TypeSig</code>  (found in <code>javafe.tc</code>).
 * If <code>type</code> is a <code>TypeName</code>, then an explicit
 * type name was given in the program text; if <code>type</code> 
 * is a <code>TypeSig</code>, then the type was inferred.
 */

public class TypeObjectDesignator extends ObjectDesignator
{
  //@ invariant type instanceof TypeName || type instanceof javafe.tc.TypeSig;
  public /*@non_null*/ Type type;


  private void postCheck() {
    Assert.notNull(type);	
  }

  //@ public represents startLoc <- (type instanceof javafe.tc.TypeSig) ? locDot : type.getStartLoc();
    public /*@ pure @*/ int getStartLoc() {
        if (!(type instanceof javafe.tc.TypeSig))
	    return type.getStartLoc();

	return locDot;
    }

  public /*@ pure @*/ Type type() { return type; }

  //* Manual maker to ensure invariant on type satisfied
  //@ requires type instanceof TypeName || type instanceof javafe.tc.TypeSig;
  //
  //@ requires locDot != javafe.util.Location.NULL;
  public static /*@non_null*/ TypeObjectDesignator make(int locDot, 
                                          /*@ non_null @*/ Type type) {
     TypeObjectDesignator result = new TypeObjectDesignator(locDot, type);
     return result;
  }


// Generated boilerplate constructors:

  //@ requires locDot != javafe.util.Location.NULL;
  //@ ensures this.locDot == locDot;
  //@ ensures this.type == type;
  protected TypeObjectDesignator(int locDot, /*@non_null*/ Type type) {
     super(locDot);
     this.type = type;
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

     if (index == 0) return this.type;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[TypeObjectDesignator"
        + " locDot = " + this.locDot
        + " type = " + this.type
        + "]";
  }

  public final int getTag() {
     return TagConstants.TYPEOBJECTDESIGNATOR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitTypeObjectDesignator(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitTypeObjectDesignator(this, o); }

  public void check() {
     super.check();
     this.type.check();
     postCheck();
  }

}
