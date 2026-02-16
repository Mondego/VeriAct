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


public class ThrowStmt extends Stmt
{
  public /*@non_null*/ Expr expr;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == expr.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.loc == loc;
  protected ThrowStmt(/*@non_null*/ Expr expr, int loc) {
     super();
     this.expr = expr;
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

     if (index == 0) return this.expr;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ThrowStmt"
        + " expr = " + this.expr
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.THROWSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitThrowStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitThrowStmt(this, o); }

  public void check() {
     super.check();
     this.expr.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ThrowStmt make(/*@non_null*/ Expr expr, int loc) {
     ThrowStmt result = new ThrowStmt(expr, loc);
     return result;
  }
}
