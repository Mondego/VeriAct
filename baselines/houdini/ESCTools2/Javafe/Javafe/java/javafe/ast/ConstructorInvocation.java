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
 * Represents an ExplicitConstructorInvocation. 
 * <p> Only occurs as the first Stmt in a ConstructorDecl. 
 * <p> Name resolution sets the decl field to the callee.
 */

public class ConstructorInvocation extends Stmt
{
  /**
   * superCall is true implies call is "super(...)",
   * superCall is false implies call is "this(...)".
   */
  public boolean superCall;


  /**
   * The enclosing instance is the object expression before a super
   * call ( <enclosingInstance>.super(...) ).  This field may be null
   * if there is no such expression.
   *
   * @note If the supertype in question is an inner class, then the
   * type checker will infer a [<C>.]this expression if no expression
   * is present and place it in this slot.  (See ThisExpr for how to
   * distinguish inferred this expressions.)<p>
   */
  public /*@nullable*/ Expr enclosingInstance;


  //@ invariant enclosingInstance != null && superCall ==> locDot != Location.NULL;
  public int locDot;


  //@ invariant locKeyword != javafe.util.Location.NULL;
  public int locKeyword;

  //@ invariant locOpenParen != javafe.util.Location.NULL;
  public int locOpenParen;


  public /*@non_null*/ ExprVec args;


  public ConstructorDecl decl;

  private void postCheck() {
    if (decl != null) {
      // Any invariants here...???
    }
  }

  /*@ public represents startLoc <- (enclosingInstance == null) 
    @		? locKeyword : enclosingInstance.getStartLoc();
    @*/
  public /*@ pure @*/ int getStartLoc() {
    if (enclosingInstance == null) return locKeyword;
    else return enclosingInstance.getStartLoc();
  }

  public /*@ pure @*/ int getEndLoc() { 
    if (args.size()<1)
      return super.getEndLoc();

    return args.elementAt(args.size()-1).getEndLoc();
  }

  //@ requires superCall ==> locDot != Location.NULL;
  //@ requires locKeyword != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  public static /*@non_null*/ ConstructorInvocation make(boolean superCall, 
                                           /*@non_null*/ Expr enclosingInstance, 
                                           int locDot, 
                                           int locKeyword, 
                                           int locOpenParen, 
                                           /*@ non_null @*/ ExprVec args) {
     ConstructorInvocation result = new ConstructorInvocation(
                                          superCall,
                                          enclosingInstance,
                                          locDot,
                                          locKeyword,
                                          locOpenParen,
                                          args);
     return result;
  }


// Generated boilerplate constructors:

  //@ requires locKeyword != javafe.util.Location.NULL;
  //@ requires locOpenParen != javafe.util.Location.NULL;
  //@ ensures this.superCall == superCall;
  //@ ensures this.enclosingInstance == enclosingInstance;
  //@ ensures this.locDot == locDot;
  //@ ensures this.locKeyword == locKeyword;
  //@ ensures this.locOpenParen == locOpenParen;
  //@ ensures this.args == args;
  protected ConstructorInvocation(boolean superCall, /*@nullable*/Expr enclosingInstance, int locDot, int locKeyword, int locOpenParen, /*@non_null*/ ExprVec args) {
     super();
     this.superCall = superCall;
     this.enclosingInstance = enclosingInstance;
     this.locDot = locDot;
     this.locKeyword = locKeyword;
     this.locOpenParen = locOpenParen;
     this.args = args;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
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

     if (index == 0) return this.enclosingInstance;
     else index--;

     ExprVec tmp_args = this.args;
     sz = (tmp_args == null ? 0 : tmp_args.size());
     if (0 <= index && index < sz)
        return tmp_args.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[ConstructorInvocation"
        + " superCall = " + this.superCall
        + " enclosingInstance = " + this.enclosingInstance
        + " locDot = " + this.locDot
        + " locKeyword = " + this.locKeyword
        + " locOpenParen = " + this.locOpenParen
        + " args = " + this.args
        + "]";
  }

  public final int getTag() {
     return TagConstants.CONSTRUCTORINVOCATION;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitConstructorInvocation(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitConstructorInvocation(this, o); }

  public void check() {
     super.check();
     if (this.enclosingInstance != null)
        this.enclosingInstance.check();
     ExprVec tmp_args = this.args;
     for(int i = 0; i < tmp_args.size(); i++)
        tmp_args.elementAt(i).check();
     postCheck();
  }

}
