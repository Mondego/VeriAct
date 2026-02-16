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


public class WhileStmt extends Stmt
{
  public /*@non_null*/ Expr expr;

  public /*@non_null*/ Stmt stmt;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  //@ invariant locGuardOpenParen != javafe.util.Location.NULL;
  public int locGuardOpenParen;


  private void postCheck() {
    int t = stmt.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == stmt.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return stmt.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locGuardOpenParen != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.stmt == stmt;
  //@ ensures this.loc == loc;
  //@ ensures this.locGuardOpenParen == locGuardOpenParen;
  protected WhileStmt(/*@non_null*/ Expr expr, /*@non_null*/ Stmt stmt, int loc, int locGuardOpenParen) {
     super();
     this.expr = expr;
     this.stmt = stmt;
     this.loc = loc;
     this.locGuardOpenParen = locGuardOpenParen;
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
     return "[WhileStmt"
        + " expr = " + this.expr
        + " stmt = " + this.stmt
        + " loc = " + this.loc
        + " locGuardOpenParen = " + this.locGuardOpenParen
        + "]";
  }

  public final int getTag() {
     return TagConstants.WHILESTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitWhileStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitWhileStmt(this, o); }

  public void check() {
     super.check();
     this.expr.check();
     this.stmt.check();
     postCheck();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locGuardOpenParen != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ WhileStmt make(/*@non_null*/ Expr expr, /*@non_null*/ Stmt stmt, int loc, int locGuardOpenParen) {
     WhileStmt result = new WhileStmt(expr, stmt, loc, locGuardOpenParen);
     return result;
  }
}
