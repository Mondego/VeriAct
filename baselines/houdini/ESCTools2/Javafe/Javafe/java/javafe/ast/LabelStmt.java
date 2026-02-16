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


public class LabelStmt extends Stmt
{
  public /*@non_null*/ Identifier label;

  public /*@non_null*/ Stmt stmt;

  //@ invariant locId != javafe.util.Location.NULL;
  public int locId;


  private void postCheck() {
    int t = stmt.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
  }
  //@ public represents startLoc <- locId;
  public /*@ pure @*/ int getStartLoc() { return locId; }
  //@ also public normal_behavior
  //@ ensures \result == stmt.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return stmt.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires locId != javafe.util.Location.NULL;
  //@ ensures this.label == label;
  //@ ensures this.stmt == stmt;
  //@ ensures this.locId == locId;
  protected LabelStmt(/*@non_null*/ Identifier label, /*@non_null*/ Stmt stmt, int locId) {
     super();
     this.label = label;
     this.stmt = stmt;
     this.locId = locId;
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

     if (index == 0) return this.label;
     else index--;

     if (index == 0) return this.stmt;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[LabelStmt"
        + " label = " + this.label
        + " stmt = " + this.stmt
        + " locId = " + this.locId
        + "]";
  }

  public final int getTag() {
     return TagConstants.LABELSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitLabelStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitLabelStmt(this, o); }

  public void check() {
     super.check();
     if (this.label == null) throw new RuntimeException();
     this.stmt.check();
     postCheck();
  }

  //@ requires locId != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ LabelStmt make(/*@non_null*/ Identifier label, /*@non_null*/ Stmt stmt, int locId) {
     LabelStmt result = new LabelStmt(label, stmt, locId);
     return result;
  }
}
