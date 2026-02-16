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
 * Represents a class literal (Type . class)
 */

public class ClassLiteral extends Expr
{
  //@ invariant type.syntax;
  public /*@non_null*/ Type type;

  //@ invariant locDot != javafe.util.Location.NULL;
  public int locDot;


  //@ public represents startLoc <- type.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return type.getStartLoc(); }

  //@ also public normal_behavior
  //@ ensures \result == locDot;
  public /*@ pure @*/ int getEndLoc() { return locDot; }


// Generated boilerplate constructors:

  //@ requires locDot != javafe.util.Location.NULL;
  //@ ensures this.type == type;
  //@ ensures this.locDot == locDot;
  protected ClassLiteral(/*@non_null*/ Type type, int locDot) {
     super();
     this.type = type;
     this.locDot = locDot;
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
     return "[ClassLiteral"
        + " type = " + this.type
        + " locDot = " + this.locDot
        + "]";
  }

  public final int getTag() {
     return TagConstants.CLASSLITERAL;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitClassLiteral(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitClassLiteral(this, o); }

  public void check() {
     super.check();
     this.type.check();
  }

  //@ requires type.syntax;
  //@ requires locDot != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ClassLiteral make(/*@non_null*/ Type type, int locDot) {
     ClassLiteral result = new ClassLiteral(type, locDot);
     return result;
  }
}
