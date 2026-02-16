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
 * Represents a SwitchLabel syntactic unit.  If expr is null, then
 * represents the "default" label, otherwise represents a case
 * label.
 *
 * <p> We infer a default: break at the end of each switch statement
 * that does not contain a default case to simplify translation.  Such
 * inferred cases will have their location equal to the location of
 * the closing bracket of the switch statement they are contained in.
 */

public class SwitchLabel extends Stmt
{
  public /*@nullable*/ Expr expr;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.expr == expr;
  //@ ensures this.loc == loc;
  protected SwitchLabel(/*@nullable*/Expr expr, int loc) {
     super();
     this.expr = expr;
     this.loc = loc;
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

     if (index == 0) return this.expr;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[SwitchLabel"
        + " expr = " + this.expr
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.SWITCHLABEL;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitSwitchLabel(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitSwitchLabel(this, o); }

  public void check() {
     super.check();
     if (this.expr != null)
        this.expr.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ SwitchLabel make(/*@nullable*/Expr expr, int loc) {
     SwitchLabel result = new SwitchLabel(expr, loc);
     return result;
  }
}
