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


public class ArrayType extends Type
{
  // We have associated locations iff elemType does:
  //@ invariant elemType.syntax == syntax;

  public /*@non_null*/ Type elemType;

  //@ invariant locOpenBracket != javafe.util.Location.NULL;
  public int locOpenBracket;


  //@ public represents startLoc <- elemType.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return elemType.getStartLoc(); }

  //@ also public normal_behavior
  //@ ensures \result == elemType.getEndLoc();
  public /*@ pure @*/ int getEndLoc() {   return Location.inc(locOpenBracket, 1);  }

  // Generate this manually to add condition about syntax:
  //@ requires locOpenBracket != javafe.util.Location.NULL;
  //@ ensures elemType.syntax ==> \result.syntax;
  public static /*@non_null*/ ArrayType make(/*@ non_null @*/ Type elemType,
                               int locOpenBracket) {
     ArrayType result = new ArrayType(null, elemType, locOpenBracket);
     // Can't fix this since elemType is *not* injective:
     //@ assume (\forall ArrayType a; a.elemType != result);
     return result;
  }

  //@ requires locOpenBracket != javafe.util.Location.NULL;
  //@ ensures elemType.syntax ==> \result.syntax;
  public static /*@non_null*/ ArrayType make(TypeModifierPragmaVec tmodifiers,
			       /*@ non_null @*/ Type elemType,
                               int locOpenBracket) {
	ArrayType at = ArrayType.make(elemType, locOpenBracket);
	at.tmodifiers = tmodifiers;
	return at;
  }


// Generated boilerplate constructors:

  //@ requires locOpenBracket != javafe.util.Location.NULL;
  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.elemType == elemType;
  //@ ensures this.locOpenBracket == locOpenBracket;
  protected ArrayType(/*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ Type elemType, int locOpenBracket) {
     super(tmodifiers);
     this.elemType = elemType;
     this.locOpenBracket = locOpenBracket;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null) sz += tmp_tmodifiers.size();
     return sz + 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     sz = (tmp_tmodifiers == null ? 0 : tmp_tmodifiers.size());
     if (0 <= index && index < sz)
        return tmp_tmodifiers.elementAt(index);
     else index -= sz;

     if (index == 0) return this.elemType;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ArrayType"
        + " tmodifiers = " + this.tmodifiers
        + " elemType = " + this.elemType
        + " locOpenBracket = " + this.locOpenBracket
        + "]";
  }

  public final int getTag() {
     return TagConstants.ARRAYTYPE;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitArrayType(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitArrayType(this, o); }

  public void check() {
     super.check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     this.elemType.check();
  }

}
