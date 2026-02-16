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
 * Represents an Expression.
 *
 * We define Expr as a subclass of VarInit to yield a more compact
 * representation. An alternative is for VarInit to contain a pointer
 * to an Expr, which would result in a number of additional pointers,
 * in particular for array initializers.
 */

public abstract class Expr extends VarInit
{ 

// Generated boilerplate constructors:

   protected Expr() {
      super();
   }
   public void check() {
      super.check();
   }

}
