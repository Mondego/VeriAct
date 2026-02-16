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


public class SynchronizeStmt extends Stmt
{
  public /*@non_null*/ Expr expr;

  public /*@non_null*/ BlockStmt stmt;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  //@ invariant locOpenParen != javafe.util.Location.NULL;
  public int locOpenParen;


  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == stmt.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return stmt.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.stmt == stmt;
  //@ ensures this.loc == loc;
  //@ ensures this.locOpenParen == locOpenParen;
  protected SynchronizeStmt(/*@non_null*/ Expr expr, /*@non_null*/ BlockStmt stmt, int loc, int locOpenParen) {
     super();
     this.expr = expr;
     this.stmt = stmt;
     this.loc = loc;
     this.locOpenParen = locOpenParen;
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

     if (index == 0) return this.expr;
     else index--;

     if (index == 0) return this.stmt;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[SynchronizeStmt"
        + " expr = " + this.expr
        + " stmt = " + this.stmt
        + " loc = " + this.loc
        + " locOpenParen = " + this.locOpenParen
        + "]";
  }

  public final int getTag() {
     return TagConstants.SYNCHRONIZESTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitSynchronizeStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitSynchronizeStmt(this, o); }

  public void check() {
     super.check();
     this.expr.check();
     this.stmt.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ SynchronizeStmt make(/*@non_null*/ Expr expr, /*@non_null*/ BlockStmt stmt, int loc, int locOpenParen) {
     SynchronizeStmt result = new SynchronizeStmt(expr, stmt, loc, locOpenParen);
     return result;
  }
}
