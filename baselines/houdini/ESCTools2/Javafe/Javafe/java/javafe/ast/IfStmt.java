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


public class IfStmt extends Stmt
{
  public /*@non_null*/ Expr expr;

  public /*@non_null*/ Stmt thn;

  public /*@non_null*/ Stmt els;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  private void postCheck() {
    int t = thn.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL 	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
    t = els.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  public /*@ pure @*/ int getEndLoc() { 
    int thenL = thn.getEndLoc();
    int elseL = els.getEndLoc();
    return Math.max(thenL, elseL);
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.thn == thn;
  //@ ensures this.els == els;
  //@ ensures this.loc == loc;
  protected IfStmt(/*@non_null*/ Expr expr, /*@non_null*/ Stmt thn, /*@non_null*/ Stmt els, int loc) {
     super();
     this.expr = expr;
     this.thn = thn;
     this.els = els;
     this.loc = loc;
  }

// Generated boilerplate methods:

  public final int childCount() {
     return 3;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.expr;
     else index--;

     if (index == 0) return this.thn;
     else index--;

     if (index == 0) return this.els;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[IfStmt"
        + " expr = " + this.expr
        + " thn = " + this.thn
        + " els = " + this.els
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.IFSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitIfStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitIfStmt(this, o); }

  public void check() {
     super.check();
     this.expr.check();
     this.thn.check();
     this.els.check();
     postCheck();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ IfStmt make(/*@non_null*/ Expr expr, /*@non_null*/ Stmt thn, /*@non_null*/ Stmt els, int loc) {
     IfStmt result = new IfStmt(expr, thn, els, loc);
     return result;
  }
}
