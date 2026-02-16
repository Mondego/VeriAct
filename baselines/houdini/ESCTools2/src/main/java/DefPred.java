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


public class DefPred extends ASTNode
{
    public /*@ non_null @*/ Identifier predId;

    public /*@ non_null @*/ GenericVarDeclVec args;

    public /*@ non_null @*/ Expr body;


    //@ public represents startLoc <- body.getStartLoc();
    public /*@ pure @*/ int getStartLoc() { return body.getStartLoc(); }


// Generated boilerplate constructors:

    //@ ensures this.predId == predId;
    //@ ensures this.args == args;
    //@ ensures this.body == body;
    protected DefPred(/*@ non_null @*/ Identifier predId, /*@ non_null @*/ GenericVarDeclVec args, /*@ non_null @*/ Expr body) {
       this.predId = predId;
       this.args = args;
       this.body = body;
    }

// Generated boilerplate methods:

    public final int childCount() {
       int sz = 0;
       if (this.args != null) sz += this.args.size();
       return sz + 2;
    }

    public final /*@ non_null */ Object childAt(int index) {
            /*throws IndexOutOfBoundsException*/
       if (index < 0)
          throw new IndexOutOfBoundsException("AST child index " + index);
       int indexPre = index;

       int sz;

       if (index == 0) return this.predId;
       else index--;

       sz = (this.args == null ? 0 : this.args.size());
       if (0 <= index && index < sz)
          return this.args.elementAt(index);
       else index -= sz;

       if (index == 0) return this.body;
       else index--;

       throw new IndexOutOfBoundsException("AST child index " + indexPre);
    }   //@ nowarn Exception;

    public final /*@non_null*/String toString() {
       return "[DefPred"
          + " predId = " + this.predId
          + " args = " + this.args
          + " body = " + this.body
          + "]";
    }

    public final int getTag() {
       return TagConstants.DEFPRED;
    }

    public final void accept(/*@ non_null */ javafe.ast.Visitor v) { 
        if (v instanceof Visitor) ((Visitor)v).visitDefPred(this);
    }

    public final /*@ non_null */ Object accept(/*@ non_null */ javafe.ast.VisitorArgResult v, /*@ nullable */ Object o) { 
        if (v instanceof VisitorArgResult) return ((VisitorArgResult)v).visitDefPred(this, o); else return null;
    }

    public void check() {
       this.predId.check();
       for(int i = 0; i < this.args.size(); i++)
          this.args.elementAt(i).check();
       this.body.check();
    }

    //@ ensures \result != null;
    public static /*@ non_null */ DefPred make(/*@ non_null @*/ Identifier predId, /*@ non_null @*/ GenericVarDeclVec args, /*@ non_null @*/ Expr body) {
       DefPred result = new DefPred(predId, args, body);
       return result;
    }
}
