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

/** Represents all variable declarations, including field declarations,
 *  local variables and formal arguments.
 *
 *  We 'unroll' field and variable decls, 
 *  so "<code>int x,y;</code>" becomes "<code>int x; int y;</code>".
 *  This unrolling can be detected by looking for sequential
 *  <code>VarDecl</code>s whose <code>Type</code> fields have the same
 *  starting location.
 */

public abstract class GenericVarDecl extends ASTNode
{
  public int modifiers;

  public /*@nullable*/ ModifierPragmaVec pmodifiers;

  public /*@non_null*/ Identifier id;

  public /*@non_null*/ Type type;

  public int locId;


  //xxx  public boolean isInternal() { return locId == javafe.util.Location.NULL; }

  //@ public represents startLoc <- type.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return type.getStartLoc(); }
  //@ also public normal_behavior
  //@ ensures \result == type.getEndLoc();
  public /*@ pure @*/ int getEndLoc() { return type.getEndLoc(); }
  public int getModifiers() { return modifiers; }
  public void setModifiers(int m) { modifiers = m; }

  protected GenericVarDecl(int modifiers, ModifierPragmaVec pmodifiers, /*@ non_null @*/ Identifier id, /*@ non_null @*/ Type type) {
     super();
     this.modifiers = modifiers;
     this.pmodifiers = pmodifiers;
     this.id = id;
     this.type = type;
     this.locId = Location.NULL;
  }


// Generated boilerplate constructors:

  //@ ensures this.modifiers == modifiers;
  //@ ensures this.pmodifiers == pmodifiers;
  //@ ensures this.id == id;
  //@ ensures this.type == type;
  //@ ensures this.locId == locId;
  protected GenericVarDecl(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@non_null*/ Identifier id, /*@non_null*/ Type type, int locId) {
     super();
     this.modifiers = modifiers;
     this.pmodifiers = pmodifiers;
     this.id = id;
     this.type = type;
     this.locId = locId;
  }
  public void check() {
     super.check();
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null)
        for(int i = 0; i < tmp_pmodifiers.size(); i++)
           tmp_pmodifiers.elementAt(i).check();
     if (this.id == null) throw new RuntimeException();
     this.type.check();
  }

}
