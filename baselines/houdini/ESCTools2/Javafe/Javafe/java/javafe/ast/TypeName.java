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


public class TypeName extends Type
{
  // We always have associated locations:
  //@ invariant syntax;

  public /*@non_null*/ Name name;

  //@ public represents startLoc <- name.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return name.getStartLoc(); }
  //@ also public normal_behavior
  //@ ensures \result == name.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return name.getEndLoc(); }

  // overloaded constructor for type names that
  // do not have any type modifiers
  static public /*@ non_null */ TypeName make(/*@ non_null @*/ Name name) {
    return TypeName.make(null, name);
  }



// Generated boilerplate constructors:

  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.name == name;
  protected TypeName(/*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ Name name) {
     super(tmodifiers);
     this.name = name;
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

     if (index == 0) return this.name;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[TypeName"
        + " tmodifiers = " + this.tmodifiers
        + " name = " + this.name
        + "]";
  }

  public final int getTag() {
     return TagConstants.TYPENAME;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitTypeName(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitTypeName(this, o); }

  public void check() {
     super.check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     this.name.check();
  }

  //@ ensures \result != null;
  public static /*@non_null*/ TypeName make(/*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ Name name) {
     TypeName result = new TypeName(tmodifiers, name);
     return result;
  }
}
