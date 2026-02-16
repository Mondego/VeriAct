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


public class TryFinallyStmt extends Stmt
{
  // @note locTry might will be shared with tryClause.locTry if
  // tryClause is a TryCatchStmt
  public /*@non_null*/ Stmt tryClause;

  public /*@non_null*/ Stmt finallyClause;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  //@ invariant locFinally != javafe.util.Location.NULL;
  public int locFinally;


  private void postCheck() {
    int t = tryClause.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
    t = finallyClause.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == finallyClause.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return finallyClause.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locFinally != javafe.util.Location.NULL;
  //@ ensures this.tryClause == tryClause;
  //@ ensures this.finallyClause == finallyClause;
  //@ ensures this.loc == loc;
  //@ ensures this.locFinally == locFinally;
  protected TryFinallyStmt(/*@non_null*/ Stmt tryClause, /*@non_null*/ Stmt finallyClause, int loc, int locFinally) {
     super();
     this.tryClause = tryClause;
     this.finallyClause = finallyClause;
     this.loc = loc;
     this.locFinally = locFinally;
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

     if (index == 0) return this.tryClause;
     else index--;

     if (index == 0) return this.finallyClause;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[TryFinallyStmt"
        + " tryClause = " + this.tryClause
        + " finallyClause = " + this.finallyClause
        + " loc = " + this.loc
        + " locFinally = " + this.locFinally
        + "]";
  }

  public final int getTag() {
     return TagConstants.TRYFINALLYSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitTryFinallyStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitTryFinallyStmt(this, o); }

  public void check() {
     super.check();
     this.tryClause.check();
     this.finallyClause.check();
     postCheck();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locFinally != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ TryFinallyStmt make(/*@non_null*/ Stmt tryClause, /*@non_null*/ Stmt finallyClause, int loc, int locFinally) {
     TryFinallyStmt result = new TryFinallyStmt(tryClause, finallyClause, loc, locFinally);
     return result;
  }
}
