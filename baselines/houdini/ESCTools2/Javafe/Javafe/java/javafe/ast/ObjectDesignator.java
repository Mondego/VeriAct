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
 * Designates the object or type used for a field or method access.
 * Subclasses represent "expr.", "type.", or "super."
 */

public abstract class ObjectDesignator extends ASTNode
{
  //@ invariant locDot != javafe.util.Location.NULL;
  public int locDot;


  //@ also public normal_behavior
  //@ ensures \result == locDot;
    public /*@ pure @*/ int getEndLoc() { return locDot; }
    abstract public Type type();


// Generated boilerplate constructors:

  //@ requires locDot != javafe.util.Location.NULL;
  //@ ensures this.locDot == locDot;
  protected ObjectDesignator(int locDot) {
     super();
     this.locDot = locDot;
  }
  public void check() {
     super.check();
  }

}
