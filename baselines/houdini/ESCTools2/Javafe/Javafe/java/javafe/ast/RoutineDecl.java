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

/** Represents both MethodDeclarations and ConstructorDeclarations.
 */

public abstract class RoutineDecl extends ASTNode implements TypeDeclElem
{
  //@ invariant hasParent ==> parent != null;
  public /*@nullable*/TypeDecl parent;

  public boolean binaryArgNames = false; // true if the ids of formal
	// arguments are binary manufactured names instead of source code names

  public boolean implicit = false;
  public boolean specOnly = false;
  public TypeNameVec originalRaises = null;

  public int modifiers;

  public /*@nullable*/ ModifierPragmaVec pmodifiers;

  public /*@nullable*/ TypeModifierPragmaVec tmodifiers;

  public /*@non_null*/ FormalParaDeclVec args;

  public /*@non_null*/ TypeNameVec raises;

  public /*@nullable*/ BlockStmt body;

  public int locOpenBrace;

  //@ invariant body != null ==> locOpenBrace != Location.NULL;
  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  //@ invariant locId != javafe.util.Location.NULL;
  public int locId;


  // The "locThrowsKeyword" field should be used only if "raises" denotes
  // a nonempty list, in which case "locThrowsKeyword" is not "Location.NULL"
  public int locThrowsKeyword;


  private void postCheck() {
    // Check invariants on modifiers...

    for(int i = 0; i < args.size(); i++)
      // Check invariants on modifiers...
      /*skip*/;
  }

  public TypeDecl getParent() { return parent; }
  public void setParent(/*@non_null*/ TypeDecl p) { parent = p; }

  public int getModifiers() { return modifiers; }
  public void setModifiers(int m) { modifiers = m; }
  public ModifierPragmaVec getPModifiers() { return pmodifiers; }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }

  abstract public Identifier id();

  public /*@ pure @*/ int getEndLoc() { 
      if (body == null)
	  return super.getEndLoc();

      return body.getEndLoc();
  }
  private Type[] argtypes = null;
  public Type[] argTypes() {
	if (argtypes != null) return argtypes;
	argtypes = new Type[args.size()];
	for (int i=0; i<args.size(); ++i) {
		argtypes[i] = args.elementAt(i).type;
	}
	return argtypes;
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locId != javafe.util.Location.NULL;
  //@ ensures this.modifiers == modifiers;
  //@ ensures this.pmodifiers == pmodifiers;
  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.args == args;
  //@ ensures this.raises == raises;
  //@ ensures this.body == body;
  //@ ensures this.locOpenBrace == locOpenBrace;
  //@ ensures this.loc == loc;
  //@ ensures this.locId == locId;
  //@ ensures this.locThrowsKeyword == locThrowsKeyword;
  protected RoutineDecl(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ FormalParaDeclVec args, /*@non_null*/ TypeNameVec raises, /*@nullable*/BlockStmt body, int locOpenBrace, int loc, int locId, int locThrowsKeyword) {
     super();
     this.modifiers = modifiers;
     this.pmodifiers = pmodifiers;
     this.tmodifiers = tmodifiers;
     this.args = args;
     this.raises = raises;
     this.body = body;
     this.locOpenBrace = locOpenBrace;
     this.loc = loc;
     this.locId = locId;
     this.locThrowsKeyword = locThrowsKeyword;
  }
  public void check() {
     super.check();
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null)
        for(int i = 0; i < tmp_pmodifiers.size(); i++)
           tmp_pmodifiers.elementAt(i).check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     FormalParaDeclVec tmp_args = this.args;
     for(int i = 0; i < tmp_args.size(); i++)
        tmp_args.elementAt(i).check();
     TypeNameVec tmp_raises = this.raises;
     for(int i = 0; i < tmp_raises.size(); i++)
        tmp_raises.elementAt(i).check();
     if (this.body != null)
        this.body.check();
     postCheck();
  }

}
