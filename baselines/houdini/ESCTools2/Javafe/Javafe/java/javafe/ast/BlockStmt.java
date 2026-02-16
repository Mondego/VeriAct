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


public class BlockStmt extends GenericBlockStmt
{
  private void postCheck() {
    for(int i = 0; i < stmts.size(); i++) {
      int t = stmts.elementAt(i).getTag();
      Assert.notFalse(t != TagConstants.SWITCHLABEL);	//@ nowarn Pre;
      Assert.notFalse(i == 0				//@ nowarn Pre;
	|| t != TagConstants.CONSTRUCTORINVOCATION);
    }
  }
  //@ public represents startLoc <- locOpenBrace;
  public /*@ pure @*/ int getStartLoc() { return locOpenBrace; }


// Generated boilerplate constructors:

  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures this.stmts == stmts;
  //@ ensures this.locOpenBrace == locOpenBrace;
  //@ ensures this.locCloseBrace == locCloseBrace;
  protected BlockStmt(/*@non_null*/ StmtVec stmts, int locOpenBrace, int locCloseBrace) {
     super(stmts, locOpenBrace, locCloseBrace);
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     StmtVec tmp_stmts = this.stmts;
     if (tmp_stmts != null) sz += tmp_stmts.size();
     return sz + 0;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     StmtVec tmp_stmts = this.stmts;
     sz = (tmp_stmts == null ? 0 : tmp_stmts.size());
     if (0 <= index && index < sz)
        return tmp_stmts.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[BlockStmt"
        + " stmts = " + this.stmts
        + " locOpenBrace = " + this.locOpenBrace
        + " locCloseBrace = " + this.locCloseBrace
        + "]";
  }

  public final int getTag() {
     return TagConstants.BLOCKSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitBlockStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitBlockStmt(this, o); }

  public void check() {
     super.check();
     StmtVec tmp_stmts = this.stmts;
     for(int i = 0; i < tmp_stmts.size(); i++)
        tmp_stmts.elementAt(i).check();
     postCheck();
  }

  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ BlockStmt make(/*@non_null*/ StmtVec stmts, int locOpenBrace, int locCloseBrace) {
     BlockStmt result = new BlockStmt(stmts, locOpenBrace, locCloseBrace);
     return result;
  }
}
