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
 * Represents a try-catch statement.
 */

public class TryCatchStmt extends Stmt
{
  public /*@non_null*/ Stmt tryClause;

  public /*@non_null*/ CatchClauseVec catchClauses;

  //@ invariant loc != javafe.util.Location.NULL;
  public int loc;


  private void postCheck() {
    int t = tryClause.getTag();
    Assert.notFalse(t != TagConstants.SWITCHLABEL	//@ nowarn Pre;
		    && t != TagConstants.CONSTRUCTORINVOCATION
		    && t != TagConstants.VARDECLSTMT);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  public /*@ pure @*/ int getEndLoc() { 
    if (catchClauses == null || catchClauses.size()<1)
      return tryClause.getEndLoc();

    return catchClauses.elementAt(catchClauses.size()-1).getEndLoc();
  }


// Generated boilerplate constructors:

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures this.tryClause == tryClause;
  //@ ensures this.catchClauses == catchClauses;
  //@ ensures this.loc == loc;
  protected TryCatchStmt(/*@non_null*/ Stmt tryClause, /*@non_null*/ CatchClauseVec catchClauses, int loc) {
     super();
     this.tryClause = tryClause;
     this.catchClauses = catchClauses;
     this.loc = loc;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     CatchClauseVec tmp_catchClauses = this.catchClauses;
     if (tmp_catchClauses != null) sz += tmp_catchClauses.size();
     return sz + 1;
  }

  public final /*@non_null*/ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.tryClause;
     else index--;

     CatchClauseVec tmp_catchClauses = this.catchClauses;
     sz = (tmp_catchClauses == null ? 0 : tmp_catchClauses.size());
     if (0 <= index && index < sz)
        return tmp_catchClauses.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[TryCatchStmt"
        + " tryClause = " + this.tryClause
        + " catchClauses = " + this.catchClauses
        + " loc = " + this.loc
        + "]";
  }

  public final int getTag() {
     return TagConstants.TRYCATCHSTMT;
  }

  public final void accept(/*@non_null*/ Visitor v) { v.visitTryCatchStmt(this); }

  public final /*@non_null*/Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o) {return v.visitTryCatchStmt(this, o); }

  public void check() {
     super.check();
     this.tryClause.check();
     CatchClauseVec tmp_catchClauses = this.catchClauses;
     for(int i = 0; i < tmp_catchClauses.size(); i++)
        tmp_catchClauses.elementAt(i).check();
     postCheck();
  }

  //@ requires loc != javafe.util.Location.NULL;
  //@ ensures \result != null;
  public static /*@non_null*/ TryCatchStmt make(/*@non_null*/ Stmt tryClause, /*@non_null*/ CatchClauseVec catchClauses, int loc) {
     TryCatchStmt result = new TryCatchStmt(tryClause, catchClauses, loc);
     return result;
  }
}
