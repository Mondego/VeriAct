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


public class OnDemandImportDecl extends ImportDecl
{
  public /*@non_null*/ Name pkgName;



// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.loc == loc;
  //@ ensures this.pkgName == pkgName;
  protected OnDemandImportDecl(int loc, /*@non_null*/ Name pkgName) {
     super(loc);
     this.pkgName = pkgName;
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

     if (index == 0) return this.pkgName;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[OnDemandImportDecl"
        + " loc = " + this.loc
        + " pkgName = " + this.pkgName
        + "]";
  }

  public final int getTag() {
     return TagConstants.ONDEMANDIMPORTDECL;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitOnDemandImportDecl(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitOnDemandImportDecl(this, o); }

  public void check() {
     super.check();
     this.pkgName.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ OnDemandImportDecl make(int loc, /*@non_null*/ Name pkgName) {
     OnDemandImportDecl result = new OnDemandImportDecl(loc, pkgName);
     return result;
  }
}
