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

public abstract class Visitor extends javafe.ast.Visitor {
  public abstract void visitAnOverview(/*@non_null*/ AnOverview x);

  public abstract void visitGCExpr(/*@non_null*/ GCExpr x);

  public void visitNaryExpr(/*@non_null*/ NaryExpr x) {
    visitGCExpr(x);
  }

  public void visitQuantifiedExpr(/*@non_null*/ QuantifiedExpr x) {
    visitGCExpr(x);
  }

  public void visitGeneralizedQuantifiedExpr(/*@non_null*/ GeneralizedQuantifiedExpr x) {
    visitGCExpr(x);
  }

  public void visitNumericalQuantifiedExpr(/*@non_null*/ NumericalQuantifiedExpr x) {
    visitGCExpr(x);
  }

  public void visitSubstExpr(/*@non_null*/ SubstExpr x) {
    visitGCExpr(x);
  }

  public void visitTypeExpr(/*@non_null*/ TypeExpr x) {
    visitGCExpr(x);
  }

  public void visitLabelExpr(/*@non_null*/ LabelExpr x) {
    visitGCExpr(x);
  }

  public abstract void visitWildRefExpr(/*@non_null*/ WildRefExpr x);

  public abstract void visitGuardExpr(/*@non_null*/ GuardExpr x);

  public abstract void visitResExpr(/*@non_null*/ ResExpr x);

  public abstract void visitSetCompExpr(/*@non_null*/ SetCompExpr x);

  public abstract void visitLockSetExpr(/*@non_null*/ LockSetExpr x);

  public abstract void visitEverythingExpr(/*@non_null*/ EverythingExpr x);

  public abstract void visitNothingExpr(/*@non_null*/ NothingExpr x);

  public abstract void visitNotSpecifiedExpr(/*@non_null*/ NotSpecifiedExpr x);

  public abstract void visitNotModifiedExpr(/*@non_null*/ NotModifiedExpr x);

  public abstract void visitArrayRangeRefExpr(/*@non_null*/ ArrayRangeRefExpr x);

  public abstract void visitDefPredLetExpr(/*@non_null*/ DefPredLetExpr x);

  public abstract void visitDefPredApplExpr(/*@non_null*/ DefPredApplExpr x);

  public abstract void visitEscPrimitiveType(/*@non_null*/ EscPrimitiveType x);

  public abstract void visitGuardedCmd(/*@non_null*/ GuardedCmd x);

  public void visitSimpleCmd(/*@non_null*/ SimpleCmd x) {
    visitGuardedCmd(x);
  }

  public void visitExprCmd(/*@non_null*/ ExprCmd x) {
    visitGuardedCmd(x);
  }

  public void visitAssignCmd(/*@non_null*/ AssignCmd x) {
    visitGuardedCmd(x);
  }

  public void visitGetsCmd(/*@non_null*/ GetsCmd x) {
    visitAssignCmd(x);
  }

  public void visitSubGetsCmd(/*@non_null*/ SubGetsCmd x) {
    visitAssignCmd(x);
  }

  public void visitSubSubGetsCmd(/*@non_null*/ SubSubGetsCmd x) {
    visitAssignCmd(x);
  }

  public void visitRestoreFromCmd(/*@non_null*/ RestoreFromCmd x) {
    visitAssignCmd(x);
  }

  public void visitVarInCmd(/*@non_null*/ VarInCmd x) {
    visitGuardedCmd(x);
  }

  public void visitDynInstCmd(/*@non_null*/ DynInstCmd x) {
    visitGuardedCmd(x);
  }

  public void visitSeqCmd(/*@non_null*/ SeqCmd x) {
    visitGuardedCmd(x);
  }

  public abstract void visitDecreasesInfo(/*@non_null*/ DecreasesInfo x);

  public void visitLoopCmd(/*@non_null*/ LoopCmd x) {
    visitGuardedCmd(x);
  }

  public void visitCmdCmdCmd(/*@non_null*/ CmdCmdCmd x) {
    visitGuardedCmd(x);
  }

  public void visitCall(/*@non_null*/ Call x) {
    visitGuardedCmd(x);
  }

  public abstract void visitExprDeclPragma(/*@non_null*/ ExprDeclPragma x);

  public abstract void visitIdExprDeclPragma(/*@non_null*/ IdExprDeclPragma x);

  public abstract void visitNamedExprDeclPragma(/*@non_null*/ NamedExprDeclPragma x);

  public abstract void visitModelDeclPragma(/*@non_null*/ ModelDeclPragma x);

  public abstract void visitDependsPragma(/*@non_null*/ DependsPragma x);

  public abstract void visitModelConstructorDeclPragma(/*@non_null*/ ModelConstructorDeclPragma x);

  public abstract void visitModelTypePragma(/*@non_null*/ ModelTypePragma x);

  public abstract void visitModelMethodDeclPragma(/*@non_null*/ ModelMethodDeclPragma x);

  public abstract void visitGhostDeclPragma(/*@non_null*/ GhostDeclPragma x);

  public abstract void visitStillDeferredDeclPragma(/*@non_null*/ StillDeferredDeclPragma x);

  public abstract void visitSimpleStmtPragma(/*@non_null*/ SimpleStmtPragma x);

  public abstract void visitIdentifierModifierPragma(/*@non_null*/ IdentifierModifierPragma x);

  public abstract void visitExprStmtPragma(/*@non_null*/ ExprStmtPragma x);

  public abstract void visitSetStmtPragma(/*@non_null*/ SetStmtPragma x);

  public abstract void visitSkolemConstantPragma(/*@non_null*/ SkolemConstantPragma x);

  public abstract void visitModelProgamModifierPragma(/*@non_null*/ ModelProgamModifierPragma x);

  public abstract void visitNestedModifierPragma(/*@non_null*/ NestedModifierPragma x);

  public abstract void visitParsedSpecs(/*@non_null*/ ParsedSpecs x);

  public abstract void visitSimpleModifierPragma(/*@non_null*/ SimpleModifierPragma x);

  public abstract void visitExprModifierPragma(/*@non_null*/ ExprModifierPragma x);

  public abstract void visitModifiesGroupPragma(/*@non_null*/ ModifiesGroupPragma x);

  public abstract void visitCondExprModifierPragma(/*@non_null*/ CondExprModifierPragma x);

  public abstract void visitMapsExprModifierPragma(/*@non_null*/ MapsExprModifierPragma x);

  public abstract void visitReachModifierPragma(/*@non_null*/ ReachModifierPragma x);

  public abstract void visitVarDeclModifierPragma(/*@non_null*/ VarDeclModifierPragma x);

  public abstract void visitVarExprModifierPragma(/*@non_null*/ VarExprModifierPragma x);

  public abstract void visitNowarnPragma(/*@non_null*/ NowarnPragma x);

  public abstract void visitImportPragma(/*@non_null*/ ImportPragma x);

  public abstract void visitRefinePragma(/*@non_null*/ RefinePragma x);

  public abstract void visitSpec(/*@non_null*/ Spec x);

  public abstract void visitCondition(/*@non_null*/ Condition x);

  public abstract void visitDefPred(/*@non_null*/ DefPred x);

}
