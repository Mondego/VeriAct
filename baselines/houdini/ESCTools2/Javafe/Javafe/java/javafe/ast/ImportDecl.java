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


/* ---------------------------------------------------------------------- */

public abstract class ImportDecl extends ASTNode
{
  //@ public represents startLoc <- loc;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  public /*@ pure @*/ int getStartLoc() { return loc; }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.loc == loc;
  protected ImportDecl(int loc) {
     super();
     this.loc = loc;
  }
  public void check() {
     super.check();
  }

}
