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
 * Represents a Name occuring before an argument list.
 * Is created by the parser, and then resolved to either an
 * InstanceMethodAccess or ClassMethodAccess by the name resolution code.
 *
 * <p> Thus for the method call "x.y()", the "x.y" part is initially 
 * represented as a MethodName, 
 * and then resolved to a InstanceMethodAccess if "x" is a variable, 
 * or resolved to a ClassMethodAccess if "x" is a type name.
 */

public class AmbiguousMethodInvocation extends Expr
{
  public /*@non_null*/ Name name;

  public /*@nullable*/ TypeModifierPragmaVec tmodifiers;

  //@ invariant locOpenParen != javafe.util.Location.NULL;
  public int locOpenParen;

  public /*@non_null*/ ExprVec args;


  //@ public represents startLoc <- name.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return name.getStartLoc(); }

  /*@ also public normal_behavior
    @ ensures \result == (args.size() < 1 ?
    @		name.getEndLoc() : args.elementAt(args.size()-1).getEndLoc());
    @*/
  public /*@ pure @*/ int getEndLoc() { 
    if (args.size()<1)
      return name.getEndLoc();

    return args.elementAt(args.size()-1).getEndLoc();
  }


// Generated boilerplate constructors:

  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures this.name == name;
  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.locOpenParen == locOpenParen;
  //@ ensures this.args == args;
  protected AmbiguousMethodInvocation(/*@non_null*/ Name name, /*@nullable*/TypeModifierPragmaVec tmodifiers, int locOpenParen, /*@non_null*/ ExprVec args) {
     super();
     this.name = name;
     this.tmodifiers = tmodifiers;
     this.locOpenParen = locOpenParen;
     this.args = args;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null) sz += tmp_tmodifiers.size();
     ExprVec tmp_args = this.args;
     if (tmp_args != null) sz += tmp_args.size();
     return sz + 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.name;
     else index--;

     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     sz = (tmp_tmodifiers == null ? 0 : tmp_tmodifiers.size());
     if (0 <= index && index < sz)
        return tmp_tmodifiers.elementAt(index);
     else index -= sz;

     ExprVec tmp_args = this.args;
     sz = (tmp_args == null ? 0 : tmp_args.size());
     if (0 <= index && index < sz)
        return tmp_args.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[AmbiguousMethodInvocation"
        + " name = " + this.name
        + " tmodifiers = " + this.tmodifiers
        + " locOpenParen = " + this.locOpenParen
        + " args = " + this.args
        + "]";
  }

  public final int getTag() {
     return TagConstants.AMBIGUOUSMETHODINVOCATION;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitAmbiguousMethodInvocation(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitAmbiguousMethodInvocation(this, o); }

  public void check() {
     super.check();
     this.name.check();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     ExprVec tmp_args = this.args;
     for(int i = 0; i < tmp_args.size(); i++)
        tmp_args.elementAt(i).check();
  }

  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ AmbiguousMethodInvocation make(/*@non_null*/ Name name, /*@nullable*/TypeModifierPragmaVec tmodifiers, int locOpenParen, /*@non_null*/ ExprVec args) {
     AmbiguousMethodInvocation result = new AmbiguousMethodInvocation(name, tmodifiers, locOpenParen, args);
     return result;
  }
}
