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


public class ClassDecl extends TypeDecl
{
  public /*@nullable*/ TypeName superClass;



  /** Set the parent pointer of the <code>TypeDeclElem</code>s
    inside the <code>this</code>. */
  private void postMake() {
    for(int i = 0; i < elems.size(); i++)
      elems.elementAt(i).setParent(this);
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locId != javafe.util.Location.NULL;
  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures this.modifiers == modifiers;
  //@ ensures this.pmodifiers == pmodifiers;
  //@ ensures this.id == id;
  //@ ensures this.superInterfaces == superInterfaces;
  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.elems == elems;
  //@ ensures this.loc == loc;
  //@ ensures this.locId == locId;
  //@ ensures this.locOpenBrace == locOpenBrace;
  //@ ensures this.locCloseBrace == locCloseBrace;
  //@ ensures this.superClass == superClass;
  protected ClassDecl(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ Identifier id, /*@non_null*/ TypeNameVec superInterfaces, /*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ TypeDeclElemVec elems, int loc, int locId, int locOpenBrace, int locCloseBrace, /*@nullable*/TypeName superClass) {
     super(modifiers, pmodifiers, id, superInterfaces, tmodifiers, elems, loc, locId, locOpenBrace, locCloseBrace);
     this.superClass = superClass;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null) sz += tmp_pmodifiers.size();
     TypeNameVec tmp_superInterfaces = this.superInterfaces;
     if (tmp_superInterfaces != null) sz += tmp_superInterfaces.size();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null) sz += tmp_tmodifiers.size();
     TypeDeclElemVec tmp_elems = this.elems;
     if (tmp_elems != null) sz += tmp_elems.size();
     return sz + 2;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     sz = (tmp_pmodifiers == null ? 0 : tmp_pmodifiers.size());
     if (0 <= index && index < sz)
        return tmp_pmodifiers.elementAt(index);
     else index -= sz;

     if (index == 0) return this.id;
     else index--;

     TypeNameVec tmp_superInterfaces = this.superInterfaces;
     sz = (tmp_superInterfaces == null ? 0 : tmp_superInterfaces.size());
     if (0 <= index && index < sz)
        return tmp_superInterfaces.elementAt(index);
     else index -= sz;

     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     sz = (tmp_tmodifiers == null ? 0 : tmp_tmodifiers.size());
     if (0 <= index && index < sz)
        return tmp_tmodifiers.elementAt(index);
     else index -= sz;

     TypeDeclElemVec tmp_elems = this.elems;
     sz = (tmp_elems == null ? 0 : tmp_elems.size());
     if (0 <= index && index < sz)
        return tmp_elems.elementAt(index);
     else index -= sz;

     if (index == 0) return this.superClass;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ClassDecl"
        + " modifiers = " + this.modifiers
        + " pmodifiers = " + this.pmodifiers
        + " id = " + this.id
        + " superInterfaces = " + this.superInterfaces
        + " tmodifiers = " + this.tmodifiers
        + " elems = " + this.elems
        + " loc = " + this.loc
        + " locId = " + this.locId
        + " locOpenBrace = " + this.locOpenBrace
        + " locCloseBrace = " + this.locCloseBrace
        + " superClass = " + this.superClass
        + "]";
  }

  public final int getTag() {
     return TagConstants.CLASSDECL;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitClassDecl(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitClassDecl(this, o); }

  public void check() {
     super.check();
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null)
        for(int i = 0; i < tmp_pmodifiers.size(); i++)
           tmp_pmodifiers.elementAt(i).check();
     if (this.id == null) throw new RuntimeException();
     TypeNameVec tmp_superInterfaces = this.superInterfaces;
     for(int i = 0; i < tmp_superInterfaces.size(); i++)
        tmp_superInterfaces.elementAt(i).check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     if (this.elems == null) throw new RuntimeException();
     if (this.superClass != null)
        this.superClass.check();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locId != javafe.util.Location.NULL;
  //@ requires locOpenBrace != javafe.util.Location.NULL;
  //@ requires locCloseBrace != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ClassDecl make(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ Identifier id, /*@non_null*/ TypeNameVec superInterfaces, /*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ TypeDeclElemVec elems, int loc, int locId, int locOpenBrace, int locCloseBrace, /*@nullable*/TypeName superClass) {
     ClassDecl result = new ClassDecl(modifiers, pmodifiers, id, superInterfaces, tmodifiers, elems, loc, locId, locOpenBrace, locCloseBrace, superClass);
     result.postMake();
     return result;
  }
}
