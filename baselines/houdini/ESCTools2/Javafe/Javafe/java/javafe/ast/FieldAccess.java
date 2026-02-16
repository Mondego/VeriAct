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
 * Represents various kinds of field access expressions. 
 * The FieldDecl is filled in by the name resolution code.
 */

public class FieldAccess extends Expr
{
  public /*@non_null*/ ObjectDesignator od;

  public /*@non_null*/ Identifier id;

  //@ invariant locId != javafe.util.Location.NULL;
  public int locId;


  //@ invariant decl == null || decl.id==id;
  public FieldDecl decl;

  private void postCheck() {
    if (decl != null) {
      Assert.notFalse(id == decl.id);
      // Any other invariants here...???
    }
  }
  //@ public represents startLoc <- (od.getStartLoc() == Location.NULL) ? locId : od.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { 
    int locOd = od.getStartLoc();
    if (locOd == Location.NULL) 
      return locId;
    else
      return locOd;
  }
  //@ also public normal_behavior
  //@ ensures \result == locId;
  public /*@ pure @*/ int getEndLoc() { return locId; }

  //@ requires locId != javafe.util.Location.NULL;
  public static /*@non_null*/ FieldAccess make(/*@ non_null @*/ ObjectDesignator od, 
                                 /*@ non_null @*/ Identifier id, 
                                 int locId) {
     FieldAccess result = new FieldAccess(
                              od,
                              id,
                              locId);
     result.decl = null; // Easier than puting an ensures on constructor
     return result;
  }


// Generated boilerplate constructors:

  //@ requires locId != javafe.util.Location.NULL;
  //@ ensures this.od == od;
  //@ ensures this.id == id;
  //@ ensures this.locId == locId;
  protected FieldAccess(/*@non_null*/ ObjectDesignator od, /*@non_null*/ Identifier id, int locId) {
     super();
     this.od = od;
     this.id = id;
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

     if (index == 0) return this.od;
     else index--;

     if (index == 0) return this.id;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[FieldAccess"
        + " od = " + this.od
        + " id = " + this.id
        + " locId = " + this.locId
        + "]";
  }

  public final int getTag() {
     return TagConstants.FIELDACCESS;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitFieldAccess(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitFieldAccess(this, o); }

  public void check() {
     super.check();
     this.od.check();
     if (this.id == null) throw new RuntimeException();
     postCheck();
  }

}
