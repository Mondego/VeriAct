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


/* ---------------------------------------------------------------------- */

/**
 * Represents a TypeDeclaration.
 * Common fields of ClassDeclarations and InterfaceDeclarations are here.
 */

public abstract class TypeDecl extends ASTNode implements TypeDeclElem
{
  /**
   * If specOnly is true, then this is only a spec.  In particular,
   * methods do not have non-empty bodies, no initializers are present
   * for fields, and no static bodies are present.
   */
  public boolean specOnly = false;

  public int modifiers;

  public /*@nullable*/ ModifierPragmaVec pmodifiers;

  public /*@non_null*/ Identifier id;

  public /*@non_null*/ TypeNameVec superInterfaces;


  public /*@nullable*/ TypeModifierPragmaVec tmodifiers;


  //* "invariant" elems.<each element>.hasParent is true
  public /*@non_null*/ TypeDeclElemVec elems;


  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  //@ invariant locId != javafe.util.Location.NULL;
  public int locId;

  //@ invariant locOpenBrace != javafe.util.Location.NULL;
  public int locOpenBrace;

  //@ invariant locCloseBrace != javafe.util.Location.NULL;
  public int locCloseBrace;


  //@ invariant hasParent ==> parent != null;
  public /*@nullable*/TypeDecl parent;

  public TypeDecl getParent() { return parent; }
  public void setParent(/*@non_null*/ TypeDecl p) { parent = p; }

  public int getModifiers() { return modifiers; }
  public void setModifiers(int m) { modifiers = m; }
  public ModifierPragmaVec getPModifiers() { return pmodifiers; }

  private void postCheck() {
    // check invariants on modifiers...
    for(int i = 0; i < elems.size(); i++) {
      Assert.notFalse(elems.elementAt(i).getParent() == this);	//@ nowarn Pre;
      for(int j = i+1; j < elems.size(); j++)
	Assert.notFalse(elems.elementAt(i) != elems.elementAt(j));  //@ nowarn Pre;
    }
  }

    /**
     * @return true iff this TypeDecl was created from a .class
     * file.
     *
     * precondition: We have already been associated with a TypeSig.
     */
    public boolean isBinary() {
        javafe.tc.TypeSig sig = javafe.tc.TypeSig.getSig(this);
	CompilationUnit cu = sig.getCompilationUnit();

	return cu.isBinary();
    }

  //@ public represents startLoc <- loc;

  public /*@ pure @*/ int getStartLoc() { return loc; }
  //@ also public normal_behavior
  //@ ensures \result == locCloseBrace;
  public /*@ pure @*/ int getEndLoc() { return locCloseBrace; }


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
  protected TypeDecl(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ Identifier id, /*@non_null*/ TypeNameVec superInterfaces, /*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ TypeDeclElemVec elems, int loc, int locId, int locOpenBrace, int locCloseBrace) {
     super();
     this.modifiers = modifiers;
     this.pmodifiers = pmodifiers;
     this.id = id;
     this.superInterfaces = superInterfaces;
     this.tmodifiers = tmodifiers;
     this.elems = elems;
     this.loc = loc;
     this.locId = locId;
     this.locOpenBrace = locOpenBrace;
     this.locCloseBrace = locCloseBrace;
  }
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
     postCheck();
  }

}
