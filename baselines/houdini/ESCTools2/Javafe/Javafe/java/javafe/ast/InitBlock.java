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


/** Represents an initializing block of code as a class member
 *
 *  We include modifiers for later extensibility to JDK 1.1, where both
 *  static and dynamic initializer blocks are allowed.
 */

public class InitBlock extends ASTNode implements TypeDeclElem
{
  //@ invariant hasParent ==> parent != null;
  public /*@nullable*/TypeDecl parent;

  public int modifiers;

  public /*@nullable*/ ModifierPragmaVec pmodifiers;

  public /*@non_null*/ BlockStmt block;


  public TypeDecl getParent() { return parent; }
  public void setParent(/*@non_null*/ TypeDecl p) { parent = p; }

  public int getModifiers() { return modifiers; }
  public void setModifiers(int m) { modifiers = m; }
  public ModifierPragmaVec getPModifiers() { return pmodifiers; }

  //@ public represents startLoc <- block.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return block.getStartLoc(); }
  public /*@ pure @*/ int getEndLoc() { return block.getEndLoc(); }


// Generated boilerplate constructors:

  //@ ensures this.modifiers == modifiers;
  //@ ensures this.pmodifiers == pmodifiers;
  //@ ensures this.block == block;
  protected InitBlock(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ BlockStmt block) {
     super();
     this.modifiers = modifiers;
     this.pmodifiers = pmodifiers;
     this.block = block;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null) sz += tmp_pmodifiers.size();
     return sz + 1;
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

     if (index == 0) return this.block;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[InitBlock"
        + " modifiers = " + this.modifiers
        + " pmodifiers = " + this.pmodifiers
        + " block = " + this.block
        + "]";
  }

  public final int getTag() {
     return TagConstants.INITBLOCK;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitInitBlock(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitInitBlock(this, o); }

  public void check() {
     super.check();
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null)
        for(int i = 0; i < tmp_pmodifiers.size(); i++)
           tmp_pmodifiers.elementAt(i).check();
     this.block.check();
  }

  //@ ensures \result != null;
  public static /*@non_null*/ InitBlock make(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ BlockStmt block) {
     InitBlock result = new InitBlock(modifiers, pmodifiers, block);
     return result;
  }
}
