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


/** Represents a ConstructorDeclaration.
 *
 *  The first Stmt in the body field should be a ConstructorInvocation. If
 *  the source code does not contain a constructor invocation, it is
 *  inserted by the parser.
 */

public class ConstructorDecl extends RoutineDecl
{

  public Identifier id() { return parent != null ? parent.id : Identifier.intern("<constructor>"); }


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
  protected ConstructorDecl(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ FormalParaDeclVec args, /*@non_null*/ TypeNameVec raises, /*@nullable*/BlockStmt body, int locOpenBrace, int loc, int locId, int locThrowsKeyword) {
     super(modifiers, pmodifiers, tmodifiers, args, raises, body, locOpenBrace, loc, locId, locThrowsKeyword);
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     ModifierPragmaVec tmp_pmodifiers = this.pmodifiers;
     if (tmp_pmodifiers != null) sz += tmp_pmodifiers.size();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null) sz += tmp_tmodifiers.size();
     FormalParaDeclVec tmp_args = this.args;
     if (tmp_args != null) sz += tmp_args.size();
     TypeNameVec tmp_raises = this.raises;
     if (tmp_raises != null) sz += tmp_raises.size();
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

     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     sz = (tmp_tmodifiers == null ? 0 : tmp_tmodifiers.size());
     if (0 <= index && index < sz)
        return tmp_tmodifiers.elementAt(index);
     else index -= sz;

     FormalParaDeclVec tmp_args = this.args;
     sz = (tmp_args == null ? 0 : tmp_args.size());
     if (0 <= index && index < sz)
        return tmp_args.elementAt(index);
     else index -= sz;

     TypeNameVec tmp_raises = this.raises;
     sz = (tmp_raises == null ? 0 : tmp_raises.size());
     if (0 <= index && index < sz)
        return tmp_raises.elementAt(index);
     else index -= sz;

     if (index == 0) return this.body;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ConstructorDecl"
        + " modifiers = " + this.modifiers
        + " pmodifiers = " + this.pmodifiers
        + " tmodifiers = " + this.tmodifiers
        + " args = " + this.args
        + " raises = " + this.raises
        + " body = " + this.body
        + " locOpenBrace = " + this.locOpenBrace
        + " loc = " + this.loc
        + " locId = " + this.locId
        + " locThrowsKeyword = " + this.locThrowsKeyword
        + "]";
  }

  public final int getTag() {
     return TagConstants.CONSTRUCTORDECL;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitConstructorDecl(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitConstructorDecl(this, o); }

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
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locId != javafe.util.Location.NULL;
  //@ requires body != null ==> locOpenBrace != Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ ConstructorDecl make(int modifiers, /*@nullable*/ModifierPragmaVec pmodifiers, /*@nullable*/TypeModifierPragmaVec tmodifiers, /*@non_null*/ FormalParaDeclVec args, /*@non_null*/ TypeNameVec raises, /*@nullable*/BlockStmt body, int locOpenBrace, int loc, int locId, int locThrowsKeyword) {
     ConstructorDecl result = new ConstructorDecl(modifiers, pmodifiers, tmodifiers, args, raises, body, locOpenBrace, loc, locId, locThrowsKeyword);
     return result;
  }
}
