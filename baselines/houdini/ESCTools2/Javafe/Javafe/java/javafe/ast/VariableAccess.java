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
 * Represents a simple name that is bound to a local variable declaration.
 * Is created only by the name resolution code, not by the parser.
 */

public class VariableAccess extends Expr
{
  public /*@non_null*/ Identifier id;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  public /*@non_null*/ GenericVarDecl decl;

  //@ invariant decl.id==id;

  private void postCheck() {
      Assert.notFalse(id == decl.id);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }

  //@ requires decl.id == id;
  //@ requires loc != javafe.util.Location.NULL;
  public static /*@non_null*/ VariableAccess make(/*@ non_null @*/ Identifier id, int loc,
                                                  /*@ non_null @*/ GenericVarDecl decl) {
	VariableAccess result = new VariableAccess(id, loc, decl);
	return result;
  }



// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.id == id;
  //@ ensures this.loc == loc;
  //@ ensures this.decl == decl;
  protected VariableAccess(/*@non_null*/ Identifier id, int loc, /*@non_null*/ GenericVarDecl decl) {
     super();
     this.id = id;
     this.loc = loc;
     this.decl = decl;
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

     if (index == 0) return this.id;
     else index--;

     if (index == 0) return this.decl;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[VariableAccess"
        + " id = " + this.id
        + " loc = " + this.loc
        + " decl = " + this.decl
        + "]";
  }

  public final int getTag() {
     return TagConstants.VARIABLEACCESS;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitVariableAccess(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitVariableAccess(this, o); }

  public void check() {
     super.check();
     if (this.id == null) throw new RuntimeException();
     this.decl.check();
     postCheck();
  }

}
