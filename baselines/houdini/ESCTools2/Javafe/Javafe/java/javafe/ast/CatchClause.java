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

public class CatchClause extends ASTNode
{
  public /*@non_null*/ FormalParaDecl arg;

  public /*@non_null*/ BlockStmt body;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == body.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return body.getEndLoc(); }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.arg == arg;
  //@ ensures this.body == body;
  //@ ensures this.loc == loc;
  protected CatchClause(/*@non_null*/ FormalParaDecl arg, /*@non_null*/ BlockStmt body, int loc) {
     super();
     this.arg = arg;
     this.body = body;
     this.loc = loc;
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

     if (index == 0) return this.arg;
     else index--;

     if (index == 0) return this.body;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[CatchClause"
        + " arg = " + this.arg
        + " body = " + this.body
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.CATCHCLAUSE;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitCatchClause(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitCatchClause(this, o); }

  public void check() {
     super.check();
     this.arg.check();
     this.body.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ CatchClause make(/*@non_null*/ FormalParaDecl arg, /*@non_null*/ BlockStmt body, int loc) {
     CatchClause result = new CatchClause(arg, body, loc);
     return result;
  }
}
