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


public abstract class StmtPragma extends Stmt
{
  /* denotes that a pragma is redundant (e.g. "assume_redundantly") */
  public boolean redundant;


  public boolean isRedundant() { return redundant; }
  public void setRedundant(boolean v) { redundant = v; }
  // when there are various synonomus tag names, tag and getTag() will be
  // a canonical value; originalTag will be the specific value.  Thus tag
  // can be used in switch statements and originalTag for printing. 
  private int originalTag;
  public StmtPragma setOriginalTag(int t) { originalTag = t; return this; }
  public int originalTag() {
    return (originalTag == 0) ? getTag() : originalTag;
  }
  protected StmtPragma() {}


// Generated boilerplate constructors:

  //@ ensures this.redundant == redundant;
  protected StmtPragma(boolean redundant) {
     super();
     this.redundant = redundant;
  }
  public void check() {
     super.check();
  }

}
