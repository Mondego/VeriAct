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


public class ArrayRefExpr extends Expr
{
  public /*@non_null*/ Expr array;

  public /*@non_null*/ Expr index;

  //@ invariant locOpenBracket != javafe.util.Location.NULL;
  public int locOpenBracket;

  //@ invariant locCloseBracket != javafe.util.Location.NULL;
  public int locCloseBracket;


  //@ public represents startLoc <- array.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return array.getStartLoc(); }
  //@ also public normal_behavior
  //@ ensures \result == locCloseBracket;
  public /*@ pure @*/ int getEndLoc() { return locCloseBracket; }


// Generated boilerplate constructors:

  //@ requires locOpenBracket != javafe.util.Location.NULL;
  //@ requires locCloseBracket != javafe.util.Location.NULL;
  //@ ensures this.array == array;
  //@ ensures this.index == index;
  //@ ensures this.locOpenBracket == locOpenBracket;
  //@ ensures this.locCloseBracket == locCloseBracket;
  protected ArrayRefExpr(/*@non_null*/ Expr array, /*@non_null*/ Expr index, int locOpenBracket, int locCloseBracket) {
     super();
     this.array = array;
     this.index = index;
     this.locOpenBracket = locOpenBracket;
     this.locCloseBracket = locCloseBracket;
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

     if (index == 0) return this.array;
     else index--;

     if (index == 0) return this.index;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ArrayRefExpr"
        + " array = " + this.array
        + " index = " + this.index
        + " locOpenBracket = " + this.locOpenBracket
        + " locCloseBracket = " + this.locCloseBracket
        + "]";
  }

  public final int getTag() {
     return TagConstants.ARRAYREFEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitArrayRefExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitArrayRefExpr(this, o); }

  public void check() {
     super.check();
     this.array.check();
     this.index.check();
  }

  //@ requires locOpenBracket != javafe.util.Location.NULL;
  //@ requires locCloseBracket != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ArrayRefExpr make(/*@non_null*/ Expr array, /*@non_null*/ Expr index, int locOpenBracket, int locCloseBracket) {
     ArrayRefExpr result = new ArrayRefExpr(array, index, locOpenBracket, locCloseBracket);
     return result;
  }
}
