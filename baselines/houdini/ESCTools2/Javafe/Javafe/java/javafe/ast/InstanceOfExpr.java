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


public class InstanceOfExpr extends Expr
{
  public /*@non_null*/ Expr expr;

  //@ invariant type.syntax;
  public /*@non_null*/ Type type;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  //@ public represents startLoc <- expr.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return expr.getStartLoc(); }
  //@ also public normal_behavior
  //@ ensures \result == type.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return type.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.type == type;
  //@ ensures this.loc == loc;
  protected InstanceOfExpr(/*@non_null*/ Expr expr, /*@non_null*/ Type type, int loc) {
     super();
     this.expr = expr;
     this.type = type;
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

     if (index == 0) return this.expr;
     else index--;

     if (index == 0) return this.type;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[InstanceOfExpr"
        + " expr = " + this.expr
        + " type = " + this.type
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.INSTANCEOFEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitInstanceOfExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitInstanceOfExpr(this, o); }

  public void check() {
     super.check();
     this.expr.check();
     this.type.check();
  }

  //@ requires type.syntax;
  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ InstanceOfExpr make(/*@non_null*/ Expr expr, /*@non_null*/ Type type, int loc) {
     InstanceOfExpr result = new InstanceOfExpr(expr, type, loc);
     return result;
  }
}
