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


public class CondExpr extends Expr
{
  public /*@non_null*/ Expr test;

  public /*@non_null*/ Expr thn;

  public /*@non_null*/ Expr els;

  //@ invariant locQuestion != javafe.util.Location.NULL;
  public int locQuestion;

  //@ invariant locColon != javafe.util.Location.NULL;
  public int locColon;


  //@ public represents startLoc <- test.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return test.getStartLoc(); }
  //@ also public normal_behavior
  //@ ensures \result == Math.max(thn.getEndLoc(), els.getEndLoc());
  public /*@ pure @*/ int getEndLoc() { 
    int thenL = thn.getEndLoc();
    int elseL = els.getEndLoc();
    return Math.max(thenL, elseL);
  }


// Generated boilerplate constructors:

  //@ requires locQuestion != javafe.util.Location.NULL;
  //@ requires locColon != javafe.util.Location.NULL;
  //@ ensures this.test == test;
  //@ ensures this.thn == thn;
  //@ ensures this.els == els;
  //@ ensures this.locQuestion == locQuestion;
  //@ ensures this.locColon == locColon;
  protected CondExpr(/*@non_null*/ Expr test, /*@non_null*/ Expr thn, /*@non_null*/ Expr els, int locQuestion, int locColon) {
     super();
     this.test = test;
     this.thn = thn;
     this.els = els;
     this.locQuestion = locQuestion;
     this.locColon = locColon;
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

     if (index == 0) return this.test;
     else index--;

     if (index == 0) return this.thn;
     else index--;

     if (index == 0) return this.els;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[CondExpr"
        + " test = " + this.test
        + " thn = " + this.thn
        + " els = " + this.els
        + " locQuestion = " + this.locQuestion
        + " locColon = " + this.locColon
        + "]";
  }

  public final int getTag() {
     return TagConstants.CONDEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitCondExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitCondExpr(this, o); }

  public void check() {
     super.check();
     this.test.check();
     this.thn.check();
     this.els.check();
  }

  //@ requires locQuestion != javafe.util.Location.NULL;
  //@ requires locColon != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ CondExpr make(/*@non_null*/ Expr test, /*@non_null*/ Expr thn, /*@non_null*/ Expr els, int locQuestion, int locColon) {
     CondExpr result = new CondExpr(test, thn, els, locQuestion, locColon);
     return result;
  }
}
