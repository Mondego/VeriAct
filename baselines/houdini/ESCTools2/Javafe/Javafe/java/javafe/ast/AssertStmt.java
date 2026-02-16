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


public class AssertStmt extends Stmt
{
  public /*@non_null*/ Expr pred;

  public /*@nullable*/ Expr label;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  public IfStmt ifStmt;

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  public /*@ pure @*/ int getEndLoc() { 
        return (label == null ? pred.getEndLoc() : label.getEndLoc());
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.pred == pred;
  //@ ensures this.label == label;
  //@ ensures this.loc == loc;
  protected AssertStmt(/*@non_null*/ Expr pred, /*@nullable*/Expr label, int loc) {
     super();
     this.pred = pred;
     this.label = label;
     this.loc = loc;
  }

// Generated boilerplate methods:

  public final int childCount() {
     return 2;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.pred;
     else index--;

     if (index == 0) return this.label;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[AssertStmt"
        + " pred = " + this.pred
        + " label = " + this.label
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.ASSERTSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitAssertStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitAssertStmt(this, o); }

  public void check() {
     super.check();
     this.pred.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ AssertStmt make(/*@non_null*/ Expr pred, /*@nullable*/Expr label, int loc) {
     AssertStmt result = new AssertStmt(pred, label, loc);
     return result;
  }
}
