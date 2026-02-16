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


public class ClassDeclStmt extends Stmt
{
  public /*@non_null*/ ClassDecl decl;


  private void postCheck() {
  // Awaiting information from Sun...
  }

  //@ public represents startLoc <- decl.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return decl.getStartLoc(); }

  //@ also public normal_behavior
  //@ ensures \result == decl.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return decl.getEndLoc(); }


// Generated boilerplate constructors:

  //@ ensures this.decl == decl;
  protected ClassDeclStmt(/*@non_null*/ ClassDecl decl) {
     super();
     this.decl = decl;
  }

// Generated boilerplate methods:

  public final int childCount() {
     return 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.decl;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ClassDeclStmt"
        + " decl = " + this.decl
        + "]";
  }

  public final int getTag() {
     return TagConstants.CLASSDECLSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitClassDeclStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitClassDeclStmt(this, o); }

  public void check() {
     super.check();
     this.decl.check();
     postCheck();
  }

  //@ ensures \result != null;
  public static /*@non_null*/ ClassDeclStmt make(/*@non_null*/ ClassDecl decl) {
     ClassDeclStmt result = new ClassDeclStmt(decl);
     return result;
  }
}
