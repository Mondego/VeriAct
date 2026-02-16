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
 * Represents a FormalParameter. 
 */

public class FormalParaDecl extends GenericVarDecl
{
  //@ invariant !isInternal();


// Generated boilerplate constructors:

   //@ ensures this.modifiers == modifiers;
   //@ ensures this.pmodifiers == pmodifiers;
   //@ ensures this.id == id;
   //@ ensures this.type == type;
   //@ ensures this.locId == locId;
   protected FormalParaDecl(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ Identifier id, /*@non_null*/ Type type, int locId) {
      super(modifiers, pmodifiers, id, type, locId);
   }

// Generated boilerplate methods:

   public final int childCount() {
      int sz = 0;
      ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
      if (tmp_pmodifiers != null) sz += tmp_pmodifiers.size();
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

      if (index == 0) return this.type;
      else index--;

      throw new IndexOutOfBoundsException("AST child index " + indexPre);
   }   //@ nowarn Exception;

   public final /*@non_null*/String toString() {
      return "[FormalParaDecl"
         + " modifiers = " + this.modifiers
         + " pmodifiers = " + this.pmodifiers
         + " id = " + this.id
         + " type = " + this.type
         + " locId = " + this.locId
         + "]";
   }

   public final int getTag() {
      return TagConstants.FORMALPARADECL;
   }

   public final void accept(/*@non_null*/ Visitor v) { v.visitFormalParaDecl(this); }

   public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitFormalParaDecl(this, o); }

   public void check() {
      super.check();
      ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
      if (tmp_pmodifiers != null)
         for(int i = 0; i < tmp_pmodifiers.size(); i++)
            tmp_pmodifiers.elementAt(i).check();
      if (this.id == null) throw new RuntimeException();
      this.type.check();
   }

   //@ ensures \result != null;
   public static /*@non_null*/ FormalParaDecl make(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ Identifier id, /*@non_null*/ Type type, int locId) {
      FormalParaDecl result = new FormalParaDecl(modifiers, pmodifiers, id, type, locId);
      return result;
   }
}
