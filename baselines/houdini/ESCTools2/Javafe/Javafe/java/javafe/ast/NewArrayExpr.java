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


public class NewArrayExpr extends Expr
{
  /**
   * The type of the elements being given zero-default values, or (if
   * an array initializer is present), the type of the array
   * initializer elements.
   *
   * <p> E.g., new int[4][3][] yields a type of int[] and new int[][][]{a, b}
   * yields a type of int[][].
   */
  //@ invariant type.syntax;
  public /*@non_null*/ Type type;


  /**
   * The array initializer, if any.  If it is present then dims will
   * contain exactly 1 element, the inferred size of the array
   * initializer.
   *
   * <p> E.g., new int[][]{7, 5} will generate a dims of {INTLIT(2)}.
   */
  //@ invariant init != null ==> dims.count==1;
  public /*@nullable*/ ArrayInit init;


  /**
   * If init is null, then holds Expr's between []'s in order.  If init
   * is not null, then holds the inferred array size.  (cf. init).
   *
   * E.g., new int[x+y][z][] will generate a dims of {<x+y>, <z>}.
   */
  //@ invariant dims.count >= 1;
  public /*@non_null*/ ExprVec dims;


  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  /**
   * The locations of the open brackets for each Expr (possibly
   * inferred if init != null) in dims.
   *
   * <p> The open bracket in front of dims[i] is located at
   * locOpenBrackets[i].
   *
   * @note locOpenBrackets may contain junk after the first
   * dims.size() entries.
   */
  /*@ invariant locOpenBrackets.length >= dims.count
    @	&& ((\forall int i; (0 <= i && i<dims.count) ==> 
			locOpenBrackets[i] != Location.NULL)); */
  public /*@non_null*/ int[] locOpenBrackets;


  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }

  public /*@ pure @*/ int getEndLoc() { 
    if (init == null) {
      if (dims.size()<1)
        Assert.fail("Invariant failure: NewArrayExpr with init & dims>0");
      return dims.elementAt(dims.size()-1).getEndLoc();
    } else return init.locCloseBrace;
  }

  private void postCheck() {
    Assert.notFalse(dims.size() >= 1);
    if (init != null)
        Assert.notFalse(dims.size()==1);
  }

  //@ requires init != null ==> dims.count==1;
  //@ requires dims.count >= 1;
  //@ requires locOpenBrackets.length >= dims.count;
  /*@ requires (\forall int i; (0 <= i && i<dims.count) ==> 
			locOpenBrackets[i] != Location.NULL); */
  //
  //@ requires type.syntax;
  //@ requires loc != javafe.util.Location.NULL;
  public static /*@non_null*/ NewArrayExpr make(/*@ non_null @*/ Type type, 
                                  /*@ non_null @*/ ExprVec dims, 
                                  ArrayInit init, 
                                  int loc, 
                                  /*@ non_null @*/ int[] locOpenBrackets) {
     NewArrayExpr result = new NewArrayExpr(
                               type,
                               init,
                               dims,
                               loc,
                               locOpenBrackets);
     return result;
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.type == type;
  //@ ensures this.init == init;
  //@ ensures this.dims == dims;
  //@ ensures this.loc == loc;
  //@ ensures this.locOpenBrackets == locOpenBrackets;
  protected NewArrayExpr(/*@non_null*/ Type type, /*@nullable*/ArrayInit init, /*@non_null*/ ExprVec dims, int loc, /*@non_null*/ int[] locOpenBrackets) {
     super();
     this.type = type;
     this.init = init;
     this.dims = dims;
     this.loc = loc;
     this.locOpenBrackets = locOpenBrackets;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     ExprVec tmp_dims = this.dims;
     if (tmp_dims != null) sz += tmp_dims.size();
     return sz + 3;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.type;
     else index--;

     if (index == 0) return this.init;
     else index--;

     ExprVec tmp_dims = this.dims;
     sz = (tmp_dims == null ? 0 : tmp_dims.size());
     if (0 <= index && index < sz)
        return tmp_dims.elementAt(index);
     else index -= sz;

     if (index == 0) return this.locOpenBrackets;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[NewArrayExpr"
        + " type = " + this.type
        + " init = " + this.init
        + " dims = " + this.dims
        + " loc = " + this.loc
        + " locOpenBrackets = " + this.locOpenBrackets
        + "]";
  }

  public final int getTag() {
     return TagConstants.NEWARRAYEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitNewArrayExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitNewArrayExpr(this, o); }

  public void check() {
     super.check();
     this.type.check();
     if (this.init != null)
        this.init.check();
     ExprVec tmp_dims = this.dims;
     for(int i = 0; i < tmp_dims.size(); i++)
        tmp_dims.elementAt(i).check();
     if (this.locOpenBrackets == null) throw new RuntimeException();
     postCheck();
  }

}
