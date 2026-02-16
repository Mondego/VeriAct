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
 * Represents an ObjectDesignator of the form "Expr . ".
 *
 * Is created both by the parser (eg for "(x).y"),
 * and by the name resolution code (eg for "x.y").
 */

public class ExprObjectDesignator extends ObjectDesignator
{
  public /*@non_null*/ Expr expr;

  //@ public represents startLoc <- expr.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return expr.getStartLoc(); }
  public Type type;
  public Type type() { return type; }


// Generated boilerplate constructors:

  //@ requires locDot != javafe.util.Location.NULL;
  //@ ensures this.locDot == locDot;
  //@ ensures this.expr == expr;
  protected ExprObjectDesignator(int locDot, /*@non_null*/ Expr expr) {
     super(locDot);
     this.expr = expr;
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
     return "[ExprObjectDesignator"
        + " locDot = " + this.locDot
        + " expr = " + this.expr
        + "]";
  }

  public final int getTag() {
     return TagConstants.EXPROBJECTDESIGNATOR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitExprObjectDesignator(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitExprObjectDesignator(this, o); }

  public void check() {
     super.check();
     this.expr.check();
  }

  //@ requires locDot != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ExprObjectDesignator make(int locDot, /*@non_null*/ Expr expr) {
     ExprObjectDesignator result = new ExprObjectDesignator(locDot, expr);
     return result;
  }
}
