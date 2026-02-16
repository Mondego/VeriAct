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


public abstract class TypeDeclElemPragma
         extends ASTNode implements TypeDeclElem
{
  /* denotes that a pragma is redundant (e.g. "invariant_redundantly") */
  public boolean redundant;


  //@ invariant hasParent ==> parent != null;
  public /*@nullable*/TypeDecl parent;

  public TypeDecl getParent() { return parent; }
  public void setParent(/*@non_null*/ TypeDecl p) { parent = p; }
  public void decorate(ModifierPragmaVec modifierPragmas) {}
  protected TypeDeclElemPragma() {}

  abstract public int getTag();
  public int getModifiers() { return 0; }
  public void setModifiers(int m) {}
  public ModifierPragmaVec getPModifiers() { return null; }
  public boolean isRedundant() { return redundant; }
  public void setRedundant(boolean v) { redundant = v; }


// Generated boilerplate constructors:

  //@ ensures this.redundant == redundant;
  protected TypeDeclElemPragma(boolean redundant) {
     super();
     this.redundant = redundant;
  }
  public void check() {
     super.check();
  }

}
