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
 * Represents various kinds of unary expressions.
 * The tag is one of the constants:
 *      ADD SUB NOT BITNOT INC DEC POSTFIXINC POSTFIXDEC
 * defined in OperatorTags.
 */

public class UnaryExpr extends Expr
{
  /*@ invariant (op == TagConstants.UNARYADD || op == TagConstants.UNARYSUB
       || op == TagConstants.NOT || op == TagConstants.BITNOT
       || op == TagConstants.INC || op == TagConstants.DEC
       || op == TagConstants.POSTFIXINC || op == TagConstants.POSTFIXDEC); */
  public int op;


  public /*@non_null*/ Expr expr;

  //@ invariant locOp != javafe.util.Location.NULL;
  public int locOp;


  public final int getTag() { return op; }

  private void postCheck() {
    boolean goodtag =
      (op == TagConstants.UNARYADD || op == TagConstants.UNARYSUB
       || op == TagConstants.NOT || op == TagConstants.BITNOT
       || op == TagConstants.INC || op == TagConstants.DEC
       || op == TagConstants.POSTFIXINC || op == TagConstants.POSTFIXDEC);
    Assert.notFalse(goodtag);
  }
  //@ public represents startLoc <- Math.min(expr.getStartLoc(), locOp);
  public /*@ pure @*/ int getStartLoc() { 
    int le = expr.getStartLoc(); 
    return le < locOp ? le : locOp;
  }

  //@ also public normal_behavior
  //@ ensures \result == Math.max(expr.getStartLoc(), locOp);
  public /*@ pure @*/ int getEndLoc() { 
    int le = expr.getEndLoc(); 
    return le < locOp ? locOp : le;
  }

  /*@ requires (op == TagConstants.UNARYADD || op == TagConstants.UNARYSUB
       || op == TagConstants.NOT || op == TagConstants.BITNOT
       || op == TagConstants.INC || op == TagConstants.DEC
       || op == TagConstants.POSTFIXINC || op == TagConstants.POSTFIXDEC); */
  //
  //@ requires locOp != javafe.util.Location.NULL;
  public static /*@non_null*/ UnaryExpr make(int op, /*@ non_null @*/ Expr expr, int locOp) {
     UnaryExpr result = new UnaryExpr(op, expr, locOp);
     return result;
  }


// Generated boilerplate constructors:

  //@ requires locOp != javafe.util.Location.NULL;
  //@ ensures this.op == op;
  //@ ensures this.expr == expr;
  //@ ensures this.locOp == locOp;
  protected UnaryExpr(int op, /*@non_null*/ Expr expr, int locOp) {
     super();
     this.op = op;
     this.expr = expr;
     this.locOp = locOp;
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
     return "[UnaryExpr"
        + " op = " + this.op
        + " expr = " + this.expr
        + " locOp = " + this.locOp
        + "]";
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitUnaryExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitUnaryExpr(this, o); }

  public void check() {
     super.check();
     this.expr.check();
     postCheck();
  }

}
