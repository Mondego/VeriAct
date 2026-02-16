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
 * Represents an ArrayInitializer. 
 */

public class ArrayInit extends VarInit
{
  public /*@non_null*/ VarInitVec elems;

  //@ invariant locOpenBrace != javafe.util.Location.NULL;
  public int locOpenBrace;

  //@ invariant locCloseBrace != javafe.util.Location.NULL;
  public int locCloseBrace;


  //@ public represents startLoc <- locOpenBrace;
  public /*@ pure @*/ int getStartLoc() { return locOpenBrace; }
  //@ also public normal_behavior
  //@ ensures \result == locCloseBrace;
  public /*@ pure @*/ int getEndLoc() { return locCloseBrace; }


// Generated boilerplate constructors:

  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures this.elems == elems;
  //@ ensures this.locOpenBrace == locOpenBrace;
  //@ ensures this.locCloseBrace == locCloseBrace;
  protected ArrayInit(/*@non_null*/ VarInitVec elems, int locOpenBrace, int locCloseBrace) {
     super();
     this.elems = elems;
     this.locOpenBrace = locOpenBrace;
     this.locCloseBrace = locCloseBrace;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     VarInitVec tmp_elems = this.elems;
     if (tmp_elems != null) sz += tmp_elems.size();
     return sz + 0;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     VarInitVec tmp_elems = this.elems;
     sz = (tmp_elems == null ? 0 : tmp_elems.size());
     if (0 <= index && index < sz)
        return tmp_elems.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ArrayInit"
        + " elems = " + this.elems
        + " locOpenBrace = " + this.locOpenBrace
        + " locCloseBrace = " + this.locCloseBrace
        + "]";
  }

  public final int getTag() {
     return TagConstants.ARRAYINIT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitArrayInit(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitArrayInit(this, o); }

  public void check() {
     super.check();
     VarInitVec tmp_elems = this.elems;
     for(int i = 0; i < tmp_elems.size(); i++)
        tmp_elems.elementAt(i).check();
  }

  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ArrayInit make(/*@non_null*/ VarInitVec elems, int locOpenBrace, int locCloseBrace) {
     ArrayInit result = new ArrayInit(elems, locOpenBrace, locCloseBrace);
     return result;
  }
}
