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


// Abstract class containing common parts of Block and SwitchBlock
// syntactic units.

public abstract class GenericBlockStmt extends Stmt
{
  public /*@non_null*/ StmtVec stmts;

  //@ invariant locOpenBrace != javafe.util.Location.NULL;
  public int locOpenBrace;

  //@ invariant locCloseBrace != javafe.util.Location.NULL;
  public int locCloseBrace;


  public /*@ pure @*/ int getEndLoc() { return locCloseBrace; }


// Generated boilerplate constructors:

  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures this.stmts == stmts;
  //@ ensures this.locOpenBrace == locOpenBrace;
  //@ ensures this.locCloseBrace == locCloseBrace;
  protected GenericBlockStmt(/*@non_null*/ StmtVec stmts, int locOpenBrace, int locCloseBrace) {
     super();
     this.stmts = stmts;
     this.locOpenBrace = locOpenBrace;
     this.locCloseBrace = locCloseBrace;
  }
  public void check() {
     super.check();
     StmtVec tmp_stmts = this.stmts;
     for(int i = 0; i < tmp_stmts.size(); i++)
        tmp_stmts.elementAt(i).check();
  }

}
