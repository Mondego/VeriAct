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


public class ParenExpr extends Expr
{
  public /*@non_null*/ Expr expr;

  //@ invariant locOpenParen != javafe.util.Location.NULL;
  public int locOpenParen;

  //@ invariant locCloseParen != javafe.util.Location.NULL;
  public int locCloseParen;


  //@ public represents startLoc <- locOpenParen;
  public /*@ pure @*/ int getStartLoc() { return locOpenParen; }
  //@ also public normal_behavior
  //@ ensures \result == locCloseParen;
  public /*@ pure @*/ int getEndLoc() { return locCloseParen; }


// Generated boilerplate constructors:

  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ requires locCloseParen != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.locOpenParen == locOpenParen;
  //@ ensures this.locCloseParen == locCloseParen;
  protected ParenExpr(/*@non_null*/ Expr expr, int locOpenParen, int locCloseParen) {
     super();
     this.expr = expr;
     this.locOpenParen = locOpenParen;
     this.locCloseParen = locCloseParen;
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
     return "[ParenExpr"
        + " expr = " + this.expr
        + " locOpenParen = " + this.locOpenParen
        + " locCloseParen = " + this.locCloseParen
        + "]";
  }

  public final int getTag() {
     return TagConstants.PARENEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitParenExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitParenExpr(this, o); }

  public void check() {
     super.check();
     this.expr.check();
  }

  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ requires locCloseParen != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ParenExpr make(/*@non_null*/ Expr expr, int locOpenParen, int locCloseParen) {
     ParenExpr result = new ParenExpr(expr, locOpenParen, locCloseParen);
     return result;
  }
}
