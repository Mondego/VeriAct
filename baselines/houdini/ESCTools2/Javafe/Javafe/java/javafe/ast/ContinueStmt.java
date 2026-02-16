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


public class ContinueStmt extends BranchStmt
{ 

// Generated boilerplate constructors:

   //@ requires loc != javafe.util.Location.NULL;
   //@ ensures this.label == label;
   //@ ensures this.loc == loc;
   protected ContinueStmt(/*@nullable*/Identifier label, int loc) {
      super(label, loc);
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

      if (index == 0) return this.label;
      else index--;

      throw new IndexOutOfBoundsException("AST child index " + indexPre);
   }   //@ nowarn Exception;

   public final /*@non_null*/String toString() {
      return "[ContinueStmt"
         + " label = " + this.label
         + " loc = " + this.loc
         + "]";
   }

   public final int getTag() {
      return TagConstants.CONTINUESTMT;
   }

   public final void accept(/*@non_null*/ Visitor v) { v.visitContinueStmt(this); }

   public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitContinueStmt(this, o); }

   public void check() {
      super.check();
   }

   //@ requires loc != javafe.util.Location.NULL;
   //@ ensures \result != null;
   public static /*@non_null*/ ContinueStmt make(/*@nullable*/Identifier label, int loc) {
      ContinueStmt result = new ContinueStmt(label, loc);
      return result;
   }
}
