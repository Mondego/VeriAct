/* $Id$
 * Copyright 2000, 2001, Compaq Computer Corporation 
 * Copyright 2006, DSRG, Concordia University
 */

package javafe.ast;

import java.io.OutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import javafe.util.Assert;
import javafe.util.Location;

public class DelegatingPrettyPrint extends PrettyPrint {

  /*@ spec_public */protected /*@ non_null */ PrettyPrint del;

  /** The contract for the default constructor does not say what del
   * is initialied to, hence strongly hinting to clients that they
   * need to initialize it explicitly. 
   */
  //@ ensures this.self == this;
  //@ ensures_redundantly this.del != null;
  protected DelegatingPrettyPrint() {
      // del needs to be non-null and PrettyPrint.inst happens to be a
      // meaningful value so use it as a filler until the client
      // explicitly initialized del.
      del = PrettyPrint.inst;
  }
  
  //@ requires del != this;
  //@ ensures this.self == self;
  //@ ensures this.del == del;
  protected DelegatingPrettyPrint(/*@ non_null */ PrettyPrint self, 
				  /*@ non_null */ PrettyPrint del) { 
    super(self); 
    this.del = del; 
  }

  //@ requires del != this;
  //@ ensures this.del == del;
  public void setDel(/*@non_null*/PrettyPrint del) {
      this.del = del;
  }
  
  public void print(/*@ non_null @*/ OutputStream o, CompilationUnit cu) {
    del.print(o, cu);
  }
  
  public void printnoln(/*@ non_null @*/ OutputStream o, int ind, TypeDecl d) {
    del.printnoln(o, ind, d);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, Stmt s) {
    del.print(o, ind, s);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, 
		    int ind, 
		    /*@ nullable */ TypeDeclElem d, 
		    /*@ non_null */ Identifier classId, 
		    boolean showBody) {
    del.print(o, ind, d, classId, showBody);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, TypeNameVec tns) {
    del.print(o, tns);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, FormalParaDeclVec fps) {
    del.print(o, ind, fps);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, ExprVec es) {
    del.print(o, ind, es);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, GenericVarDecl d) {
    del.print(o, d);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, LocalVarDecl d,
		    boolean showBody) {
    del.print(o, ind, d, showBody);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, FieldDecl d, boolean showBody) {
    del.print(o, ind, d, showBody);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, /*@ non_null */ Type t) {
    del.print(o, t);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, Name n) {
    del.print(o, n);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, VarInit e) {
    del.print(o, ind, e);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, ObjectDesignator od) {
    del.print(o, ind, od);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, /*@ non_null */ LexicalPragma lp) {
    del.print(o, lp);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, /*@ non_null */ TypeDeclElemPragma tp) {
    del.print(o, ind, tp);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, /*@ non_null */ ModifierPragma mp) {
    del.print(o, ind, mp);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, /*@ non_null */ StmtPragma sp) {
    del.print(o, ind, sp);
  }
  
  public void print(/*@ non_null @*/ OutputStream o, int ind, /*@ non_null */ TypeModifierPragma tp) {
    del.print(o, ind, tp);
  }
  
  public /*@ non_null @*/ String toString(int tag) {
    return del.toString(tag);
  }
}




