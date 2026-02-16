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


public class NewInstanceExpr extends Expr
{
  /**
   * The enclosing instance is the object expression before a new
   * call ( <enclosingInstance>.new T(...) ).  This field may be null
   * if there is no such expression.
   *
   * @note If the type in question is an inner class, then the
   * type checker will infer a [<C>.]this expression if no expression
   * is present and place it in this slot.  (See ThisExpr for how to
   * distinguish inferred this expressions.)<p>
   */
  public /*@nullable*/ Expr enclosingInstance;


  //@ invariant (enclosingInstance == null) == (locDot==Location.NULL);
  public int locDot;


  public /*@non_null*/ TypeName type;

  public /*@non_null*/ ExprVec args;


  /**
   * If the new expression includes a declaration of an inner class,
   * then "anonDecl" will be non-null.  In this case, the
   * "superclass" field of "ananDecl" will be null and
   * "superinterfaces" list of "anonDecl" will be empty.  One of
   * these fields needs to be modified during type checking depending
   * on whether "type" is a class or an interface.
   */
  public /*@nullable*/ ClassDecl anonDecl;


  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;

  //@ invariant locOpenParen != javafe.util.Location.NULL;
  public int locOpenParen;


  public ConstructorDecl decl;

  /*@ public represents startLoc <- 
    @ 		(enclosingInstance == null) ? loc : enclosingInstance.getStartLoc();
    @*/
  public /*@ pure @*/ int getStartLoc() {
    if (enclosingInstance == null) return loc;
    else return enclosingInstance.getStartLoc();
  }

  public /*@ pure @*/ int getEndLoc() {
    if (decl == null) {
      if (args.size()==0) return type.getEndLoc();
      return args.elementAt(args.size()-1).getEndLoc();
    } else return decl.getEndLoc();
  }

  //@ requires (enclosingInstance == null) == (locDot==Location.NULL);
  //
  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  public static /*@non_null*/ NewInstanceExpr make(Expr enclosingInstance, 
                                     int locDot, 
                                     /*@ non_null @*/ TypeName type, 
                                     /*@ non_null @*/ ExprVec args, 
                                     ClassDecl anonDecl, 
                                     int loc, 
                                     int locOpenParen) {
     NewInstanceExpr result = new NewInstanceExpr(
                                  enclosingInstance,
                                  locDot,
                                  type,
                                  args,
                                  anonDecl,
                                  loc,
                                  locOpenParen);
     return result;
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures this.enclosingInstance == enclosingInstance;
  //@ ensures this.locDot == locDot;
  //@ ensures this.type == type;
  //@ ensures this.args == args;
  //@ ensures this.anonDecl == anonDecl;
  //@ ensures this.loc == loc;
  //@ ensures this.locOpenParen == locOpenParen;
  protected NewInstanceExpr(/*@nullable*/Expr enclosingInstance, int locDot, /*@non_null*/ TypeName type, /*@non_null*/ ExprVec args, /*@nullable*/ClassDecl anonDecl, int loc, int locOpenParen) {
     super();
     this.enclosingInstance = enclosingInstance;
     this.locDot = locDot;
     this.type = type;
     this.args = args;
     this.anonDecl = anonDecl;
     this.loc = loc;
     this.locOpenParen = locOpenParen;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     ExprVec tmp_args = this.args;
     if (tmp_args != null) sz += tmp_args.size();
     return sz + 3;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.enclosingInstance;
     else index--;

     if (index == 0) return this.type;
     else index--;

     ExprVec tmp_args = this.args;
     sz = (tmp_args == null ? 0 : tmp_args.size());
     if (0 <= index && index < sz)
        return tmp_args.elementAt(index);
     else index -= sz;

     if (index == 0) return this.anonDecl;
     else index--;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[NewInstanceExpr"
        + " enclosingInstance = " + this.enclosingInstance
        + " locDot = " + this.locDot
        + " type = " + this.type
        + " args = " + this.args
        + " anonDecl = " + this.anonDecl
        + " loc = " + this.loc
        + " locOpenParen = " + this.locOpenParen
        + "]";
  }

  public final int getTag() {
     return TagConstants.NEWINSTANCEEXPR;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitNewInstanceExpr(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitNewInstanceExpr(this, o); }

  public void check() {
     super.check();
     if (this.enclosingInstance != null)
        this.enclosingInstance.check();
     this.type.check();
     ExprVec tmp_args = this.args;
     for(int i = 0; i < tmp_args.size(); i++)
        tmp_args.elementAt(i).check();
     if (this.anonDecl != null)
        this.anonDecl.check();
  }

}
