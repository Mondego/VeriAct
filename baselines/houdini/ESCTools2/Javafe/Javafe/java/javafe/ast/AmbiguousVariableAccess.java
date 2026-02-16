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
 * Represents a Name that occurs in an Expression position. 
 * These are name-resolved to VariableAccess,
 * ExprFieldAccess or TypeFieldAccess.
 */

public class AmbiguousVariableAccess extends Expr
{
  public /*@non_null*/ Name name;


  //@ public represents startLoc <- name.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return name.getStartLoc(); }
  //@ also public normal_behavior
  //@ ensures \result == name.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return name.getEndLoc(); }


// Generated boilerplate constructors:

  //@ ensures this.name == name;
  protected AmbiguousVariableAccess(/*@non_null*/ Name name) {
     super();
     this.name = name;
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

     if (index == 0) return this.name;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[AmbiguousVariableAccess"
        + " name = " + this.name
        + "]";
  }

  public final int getTag() {
     return TagConstants.AMBIGUOUSVARIABLEACCESS;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitAmbiguousVariableAccess(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitAmbiguousVariableAccess(this, o); }

  public void check() {
     super.check();
     this.name.check();
  }

  //@ ensures \result != null;
  public static /*@non_null*/ AmbiguousVariableAccess make(/*@non_null*/ Name name) {
     AmbiguousVariableAccess result = new AmbiguousVariableAccess(name);
     return result;
  }
}
