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
 * Represents a ObjectDesignator of the form "super . ".
 * Is created only by the parser, not by name resolution.
 */

public class SuperObjectDesignator extends ObjectDesignator
{
  //@ invariant locSuper != javafe.util.Location.NULL;
  public int locSuper;


  //@ public represents startLoc <- locSuper;
  public /*@ pure @*/ int getStartLoc() { return locSuper; }
  public Type type;
  public Type type() { return type; }


// Generated boilerplate constructors:

  //@ requires locDot != javafe.util.Location.NULL;
  //@ requires locSuper != javafe.util.Location.NULL;
  //@ ensures this.locDot == locDot;
  //@ ensures this.locSuper == locSuper;
  protected SuperObjectDesignator(int locDot, int locSuper) {
     super(locDot);
     this.locSuper = locSuper;
  }

// Generated boilerplate methods:

  public final int childCount() {
     return 0;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[SuperObjectDesignator"
        + " locDot = " + this.locDot
        + " locSuper = " + this.locSuper
        + "]";
  }

  public final int getTag() {
     return TagConstants.SUPEROBJECTDESIGNATOR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitSuperObjectDesignator(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitSuperObjectDesignator(this, o); }

  public void check() {
     super.check();
  }

  //@ requires locDot != javafe.util.Location.NULL;
  //@ requires locSuper != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ SuperObjectDesignator make(int locDot, int locSuper) {
     SuperObjectDesignator result = new SuperObjectDesignator(locDot, locSuper);
     return result;
  }
}
