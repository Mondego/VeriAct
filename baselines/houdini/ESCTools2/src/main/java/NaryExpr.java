// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package escjava.ast;

import javafe.ast.ASTDecoration;
import javafe.ast.ASTDotVisitor;
import javafe.ast.ASTNode;
import javafe.ast.AmbiguousMethodInvocation;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayInit;
import javafe.ast.ArrayRefExpr;
import javafe.ast.ArrayType;
import javafe.ast.AssertStmt;
import javafe.ast.BinaryExpr;
import javafe.ast.BlockStmt;
import javafe.ast.BranchStmt;
import javafe.ast.BreakStmt;
import javafe.ast.CastExpr;
import javafe.ast.CatchClause;
import javafe.ast.CatchClauseVec;
import javafe.ast.ClassDecl;
import javafe.ast.ClassDeclStmt;
import javafe.ast.ClassLiteral;
import javafe.ast.CompilationUnit;
import javafe.ast.CompoundName;
import javafe.ast.CondExpr;
import javafe.ast.ConstructorDecl;
import javafe.ast.ConstructorInvocation;
import javafe.ast.ContinueStmt;
import javafe.ast.DefaultVisitor;
import javafe.ast.DelegatingPrettyPrint;
import javafe.ast.DoStmt;
import javafe.ast.ErrorType;
import javafe.ast.EvalStmt;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.FieldDeclVec;
import javafe.ast.ForStmt;
import javafe.ast.FormalParaDecl;
import javafe.ast.FormalParaDeclVec;
// import javafe.ast.GeneratedTags;
import javafe.ast.GenericBlockStmt;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.IdPragma;
import javafe.ast.Identifier;
import javafe.ast.IdentifierNode;
import javafe.ast.IdentifierVec;
import javafe.ast.IfStmt;
import javafe.ast.ImportDecl;
import javafe.ast.ImportDeclVec;
import javafe.ast.InitBlock;
import javafe.ast.InstanceOfExpr;
import javafe.ast.InterfaceDecl;
import javafe.ast.JavafePrimitiveType;
import javafe.ast.LabelStmt;
import javafe.ast.LexicalPragma;
import javafe.ast.LexicalPragmaVec;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.LocalVarDeclVec;
import javafe.ast.MethodDecl;
import javafe.ast.MethodDeclVec;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.Modifiers;
import javafe.ast.Name;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.OnDemandImportDecl;
import javafe.ast.OperatorTags;
import javafe.ast.ParenExpr;
import javafe.ast.PrettyPrint;
import javafe.ast.PrimitiveType;
import javafe.ast.ReturnStmt;
import javafe.ast.RoutineDecl;
import javafe.ast.SimpleName;
import javafe.ast.SingleTypeImportDecl;
import javafe.ast.SkipStmt;
import javafe.ast.StandardPrettyPrint;
import javafe.ast.Stmt;
import javafe.ast.StmtPragma;
import javafe.ast.StmtVec;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.SwitchLabel;
import javafe.ast.SwitchStmt;
import javafe.ast.SynchronizeStmt;
// import javafe.ast.TagConstants;
import javafe.ast.ThisExpr;
import javafe.ast.ThrowStmt;
import javafe.ast.TryCatchStmt;
import javafe.ast.TryFinallyStmt;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.TypeDeclVec;
import javafe.ast.TypeModifierPragma;
import javafe.ast.TypeModifierPragmaVec;
import javafe.ast.TypeName;
import javafe.ast.TypeNameVec;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.UnaryExpr;
import javafe.ast.Util;
import javafe.ast.VarDeclStmt;
import javafe.ast.VarInit;
import javafe.ast.VarInitVec;
import javafe.ast.VariableAccess;
// import javafe.ast.Visitor;
// import javafe.ast.VisitorArgResult;
import javafe.ast.WhileStmt;

import java.util.Hashtable;
import java.util.Set;
import java.util.ArrayList;

import javafe.util.Assert;
import javafe.util.Location;
import escjava.ParsedRoutineSpecs;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit

//# TagBase javafe.tc.TagConstants.LAST_TAG + 1
//# VisitorRoot javafe.ast.Visitor


public class NaryExpr extends GCExpr
{
  public int op;

  public /*@ non_null @*/ Identifier methodName;

  public /*@ non_null @*/ ExprVec exprs;

  
  public ASTNode symbol;

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)

  public final int getTag() { return op; }

  private void postCheck() {
    boolean goodtag = 
      (
       op == TagConstants.CLASSLITERALFUNC
       || op == TagConstants.DTTFSA
       || op == TagConstants.ELEMTYPE
       || op == TagConstants.FRESH
       || op == TagConstants.MAX
       || op == TagConstants.NOWARN_OP
       || op == TagConstants.TYPEOF
       || op == TagConstants.WACK_BIGINT_MATH
       || op == TagConstants.WACK_DURATION
       || op == TagConstants.WACK_JAVA_MATH
       || op == TagConstants.WACK_NOWARN
       || op == TagConstants.WACK_SAFE_MATH
       || op == TagConstants.WACK_WORKING_SPACE
       || op == TagConstants.WARN
       || op == TagConstants.WARN_OP
       || (TagConstants.FIRSTFUNCTIONTAG <= op 
	   && op <= TagConstants.LASTFUNCTIONTAG)
       );
    Assert.notFalse(goodtag);
  }



// Generated boilerplate constructors:

  //@ ensures this.sloc == sloc;
  //@ ensures this.eloc == eloc;
  //@ ensures this.op == op;
  //@ ensures this.methodName == methodName;
  //@ ensures this.exprs == exprs;
  protected NaryExpr(int sloc, int eloc, int op, /*@ non_null @*/ Identifier methodName, /*@ non_null @*/ ExprVec exprs) {
     super(sloc, eloc);
     this.op = op;
     this.methodName = methodName;
     this.exprs = exprs;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     if (this.exprs != null) sz += this.exprs.size();
     return sz + 1;
  }

  public final /*@ non_null */ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     if (index == 0) return this.methodName;
     else index--;

     sz = (this.exprs == null ? 0 : this.exprs.size());
     if (0 <= index && index < sz)
        return this.exprs.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[NaryExpr"
        + " sloc = " + this.sloc
        + " eloc = " + this.eloc
        + " op = " + this.op
        + " methodName = " + this.methodName
        + " exprs = " + this.exprs
        + "]";
  }

  public final void accept(/*@ non_null */ javafe.ast.Visitor v) { 
    if (v instanceof Visitor) ((Visitor)v).visitNaryExpr(this);
  }

  public final /*@ non_null */ Object accept(/*@ non_null */ javafe.ast.VisitorArgResult v, /*@ nullable */ Object o) { 
    if (v instanceof VisitorArgResult) return ((VisitorArgResult)v).visitNaryExpr(this, o); else return null;
  }

  public void check() {
     super.check();
     this.methodName.check();
     for(int i = 0; i < this.exprs.size(); i++)
        this.exprs.elementAt(i).check();
     postCheck();
  }

  //@ ensures \result != null;
  public static /*@ non_null */ NaryExpr make(int sloc, int eloc, int op, /*@ non_null @*/ Identifier methodName, /*@ non_null @*/ ExprVec exprs) {
     NaryExpr result = new NaryExpr(sloc, eloc, op, methodName, exprs);
     return result;
  }
}
