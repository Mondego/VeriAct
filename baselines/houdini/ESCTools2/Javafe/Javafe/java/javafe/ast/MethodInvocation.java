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
 * Represents a MethodInvocation. 
 */

public class MethodInvocation extends Expr
{
  public /*@non_null*/ ObjectDesignator od;

  public /*@non_null*/ Identifier id;

  public /*@nullable*/ TypeModifierPragmaVec tmodifiers;

  //@ invariant locId != javafe.util.Location.NULL;
  public int locId;

  //@ invariant locOpenParen != javafe.util.Location.NULL;
  public int locOpenParen;

  public /*@non_null*/ ExprVec args;


  //@ invariant decl == null || decl.id==id;
  public MethodDecl decl;

  private void postCheck() {
    if (decl != null) {
      Assert.notFalse(id == decl.id);
      // Any other invariants here...???
    }
  }
  //@ public represents startLoc <- (od.getStartLoc() == Location.NULL) ? locId : od.getStartLoc();
  public /*@ pure @*/ int getStartLoc() {
    int loc = od.getStartLoc();
    if (loc != Location.NULL)
      return loc;
    else
      return locId;
  }
  public /*@ pure @*/ int getEndLoc() { 
    if (args.size()<1)
      return locId;

    return args.elementAt(args.size()-1).getEndLoc();
  }

  //@ requires locId != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  public static /*@non_null*/ MethodInvocation make(/*@ non_null @*/ ObjectDesignator od, 
                                      /*@ non_null @*/ Identifier id, 
                                      TypeModifierPragmaVec tmodifiers, 
                                      int locId, 
                                      int locOpenParen, 
                                      /*@ non_null @*/ ExprVec args) {
     MethodInvocation result = new MethodInvocation(
                                   od,
                                   id,
                                   tmodifiers,
                                   locId,
                                   locOpenParen,
                                   args);
     result.decl = null;  // Easier than puting an ensures on constructor
     return result;
  }


// Generated boilerplate constructors:

  //@ requires locId != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures this.od == od;
  //@ ensures this.id == id;
  //@ ensures this.tmodifiers == tmodifiers;
  //@ ensures this.locId == locId;
  //@ ensures this.locOpenParen == locOpenParen;
  //@ ensures this.args == args;
  protected MethodInvocation(/*@non_null*/ ObjectDesignator od, /*@non_null*/ Identifier id, /*@nullable*/TypeModifierPragmaVec tmodifiers, int locId, int locOpenParen, /*@non_null*/ ExprVec args) {
     super();
     this.od = od;
     this.id = id;
     this.tmodifiers = tmodifiers;
     this.locId = locId;
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
     return sz + 2;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.od;
     else index--;

     if (index == 0) return this.id;
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
     return "[MethodInvocation"
        + " od = " + this.od
        + " id = " + this.id
        + " tmodifiers = " + this.tmodifiers
        + " locId = " + this.locId
        + " locOpenParen = " + this.locOpenParen
        + " args = " + this.args
        + "]";
  }

  public final int getTag() {
     return TagConstants.METHODINVOCATION;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitMethodInvocation(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitMethodInvocation(this, o); }

  public void check() {
     super.check();
     this.od.check();
     if (this.id == null) throw new RuntimeException();
     TypeModifierPragmaVec tmp_tmodifiers = this.tmodifiers;
     if (tmp_tmodifiers != null)
        for(int i = 0; i < tmp_tmodifiers.size(); i++)
           tmp_tmodifiers.elementAt(i).check();
     ExprVec tmp_args = this.args;
     for(int i = 0; i < tmp_args.size(); i++)
        tmp_args.elementAt(i).check();
     postCheck();
  }

}
