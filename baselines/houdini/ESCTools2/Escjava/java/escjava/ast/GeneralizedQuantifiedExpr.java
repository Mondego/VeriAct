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


public class GeneralizedQuantifiedExpr extends GCExpr
{
  // Sum, Product, Max, Min
  public int quantifier;

  public /*@ non_null @*/ GenericVarDeclVec vars;

  public /*@ non_null @*/ Expr expr;

  public /*@ non_null @*/ Expr rangeExpr;

  public ExprVec nopats;


  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)

  public final int getTag() { return quantifier; }

  private void postCheck() {
    boolean goodtag =
      (quantifier == TagConstants.MIN
       || quantifier == TagConstants.PRODUCT
       || quantifier == TagConstants.MAXQUANT
       || quantifier == TagConstants.SUM);
    Assert.notFalse(goodtag);
  }


// Generated boilerplate constructors:

  //@ ensures this.sloc == sloc;
  //@ ensures this.eloc == eloc;
  //@ ensures this.quantifier == quantifier;
  //@ ensures this.vars == vars;
  //@ ensures this.expr == expr;
  //@ ensures this.rangeExpr == rangeExpr;
  //@ ensures this.nopats == nopats;
  protected GeneralizedQuantifiedExpr(int sloc, int eloc, int quantifier, /*@ non_null @*/ GenericVarDeclVec vars, /*@ non_null @*/ Expr expr, /*@ non_null @*/ Expr rangeExpr, ExprVec nopats) {
     super(sloc, eloc);
     this.quantifier = quantifier;
     this.vars = vars;
     this.expr = expr;
     this.rangeExpr = rangeExpr;
     this.nopats = nopats;
  }

// Generated boilerplate methods:

  public final int childCount() {
     int sz = 0;
     if (this.vars != null) sz += this.vars.size();
     if (this.nopats != null) sz += this.nopats.size();
     return sz + 2;
  }

  public final /*@ non_null */ Object childAt(int index) {
          /*throws IndexOutOfBoundsException*/
     if (index < 0)
        throw new IndexOutOfBoundsException("AST child index " + index);
     int indexPre = index;

     int sz;

     sz = (this.vars == null ? 0 : this.vars.size());
     if (0 <= index && index < sz)
        return this.vars.elementAt(index);
     else index -= sz;

     if (index == 0) return this.expr;
     else index--;

     if (index == 0) return this.rangeExpr;
     else index--;

     sz = (this.nopats == null ? 0 : this.nopats.size());
     if (0 <= index && index < sz)
        return this.nopats.elementAt(index);
     else index -= sz;

     throw new IndexOutOfBoundsException("AST child index " + indexPre);
  }   //@ nowarn Exception;

  public final /*@non_null*/String toString() {
     return "[GeneralizedQuantifiedExpr"
        + " sloc = " + this.sloc
        + " eloc = " + this.eloc
        + " quantifier = " + this.quantifier
        + " vars = " + this.vars
        + " expr = " + this.expr
        + " rangeExpr = " + this.rangeExpr
        + " nopats = " + this.nopats
        + "]";
  }

  public final void accept(/*@ non_null */ javafe.ast.Visitor v) { 
    if (v instanceof Visitor) ((Visitor)v).visitGeneralizedQuantifiedExpr(this);
  }

  public final /*@ non_null */ Object accept(/*@ non_null */ javafe.ast.VisitorArgResult v, /*@ nullable */ Object o) { 
    if (v instanceof VisitorArgResult) return ((VisitorArgResult)v).visitGeneralizedQuantifiedExpr(this, o); else return null;
  }

  public void check() {
     super.check();
     for(int i = 0; i < this.vars.size(); i++)
        this.vars.elementAt(i).check();
     this.expr.check();
     this.rangeExpr.check();
     if (this.nopats != null)
        for(int i = 0; i < this.nopats.size(); i++)
           this.nopats.elementAt(i).check();
     postCheck();
  }

  //@ ensures \result != null;
  public static /*@ non_null */ GeneralizedQuantifiedExpr make(int sloc, int eloc, int quantifier, /*@ non_null @*/ GenericVarDeclVec vars, /*@ non_null @*/ Expr expr, /*@ non_null @*/ Expr rangeExpr, ExprVec nopats) {
     GeneralizedQuantifiedExpr result = new GeneralizedQuantifiedExpr(sloc, eloc, quantifier, vars, expr, rangeExpr, nopats);
     return result;
  }
}
