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

public abstract class VisitorArgResult extends javafe.ast.VisitorArgResult {
  public abstract /*@non_null*/ Object visitAnOverview(/*@non_null*/ AnOverview x, Object o);

  public abstract /*@non_null*/ Object visitGCExpr(/*@non_null*/ GCExpr x, Object o);

  public /*@non_null*/ Object visitNaryExpr(/*@non_null*/ NaryExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public /*@non_null*/ Object visitQuantifiedExpr(/*@non_null*/ QuantifiedExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public /*@non_null*/ Object visitGeneralizedQuantifiedExpr(/*@non_null*/ GeneralizedQuantifiedExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public /*@non_null*/ Object visitNumericalQuantifiedExpr(/*@non_null*/ NumericalQuantifiedExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public /*@non_null*/ Object visitSubstExpr(/*@non_null*/ SubstExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public /*@non_null*/ Object visitTypeExpr(/*@non_null*/ TypeExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public /*@non_null*/ Object visitLabelExpr(/*@non_null*/ LabelExpr x, Object o) {
    return visitGCExpr(x, o);
  }

  public abstract /*@non_null*/ Object visitWildRefExpr(/*@non_null*/ WildRefExpr x, Object o);

  public abstract /*@non_null*/ Object visitGuardExpr(/*@non_null*/ GuardExpr x, Object o);

  public abstract /*@non_null*/ Object visitResExpr(/*@non_null*/ ResExpr x, Object o);

  public abstract /*@non_null*/ Object visitSetCompExpr(/*@non_null*/ SetCompExpr x, Object o);

  public abstract /*@non_null*/ Object visitLockSetExpr(/*@non_null*/ LockSetExpr x, Object o);

  public abstract /*@non_null*/ Object visitEverythingExpr(/*@non_null*/ EverythingExpr x, Object o);

  public abstract /*@non_null*/ Object visitNothingExpr(/*@non_null*/ NothingExpr x, Object o);

  public abstract /*@non_null*/ Object visitNotSpecifiedExpr(/*@non_null*/ NotSpecifiedExpr x, Object o);

  public abstract /*@non_null*/ Object visitNotModifiedExpr(/*@non_null*/ NotModifiedExpr x, Object o);

  public abstract /*@non_null*/ Object visitArrayRangeRefExpr(/*@non_null*/ ArrayRangeRefExpr x, Object o);

  public abstract /*@non_null*/ Object visitDefPredLetExpr(/*@non_null*/ DefPredLetExpr x, Object o);

  public abstract /*@non_null*/ Object visitDefPredApplExpr(/*@non_null*/ DefPredApplExpr x, Object o);

  public abstract /*@non_null*/ Object visitEscPrimitiveType(/*@non_null*/ EscPrimitiveType x, Object o);

  public abstract /*@non_null*/ Object visitGuardedCmd(/*@non_null*/ GuardedCmd x, Object o);

  public /*@non_null*/ Object visitSimpleCmd(/*@non_null*/ SimpleCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitExprCmd(/*@non_null*/ ExprCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitAssignCmd(/*@non_null*/ AssignCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitGetsCmd(/*@non_null*/ GetsCmd x, Object o) {
    return visitAssignCmd(x, o);
  }

  public /*@non_null*/ Object visitSubGetsCmd(/*@non_null*/ SubGetsCmd x, Object o) {
    return visitAssignCmd(x, o);
  }

  public /*@non_null*/ Object visitSubSubGetsCmd(/*@non_null*/ SubSubGetsCmd x, Object o) {
    return visitAssignCmd(x, o);
  }

  public /*@non_null*/ Object visitRestoreFromCmd(/*@non_null*/ RestoreFromCmd x, Object o) {
    return visitAssignCmd(x, o);
  }

  public /*@non_null*/ Object visitVarInCmd(/*@non_null*/ VarInCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitDynInstCmd(/*@non_null*/ DynInstCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitSeqCmd(/*@non_null*/ SeqCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public abstract /*@non_null*/ Object visitDecreasesInfo(/*@non_null*/ DecreasesInfo x, Object o);

  public /*@non_null*/ Object visitLoopCmd(/*@non_null*/ LoopCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitCmdCmdCmd(/*@non_null*/ CmdCmdCmd x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public /*@non_null*/ Object visitCall(/*@non_null*/ Call x, Object o) {
    return visitGuardedCmd(x, o);
  }

  public abstract /*@non_null*/ Object visitExprDeclPragma(/*@non_null*/ ExprDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitIdExprDeclPragma(/*@non_null*/ IdExprDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitNamedExprDeclPragma(/*@non_null*/ NamedExprDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitModelDeclPragma(/*@non_null*/ ModelDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitDependsPragma(/*@non_null*/ DependsPragma x, Object o);

  public abstract /*@non_null*/ Object visitModelConstructorDeclPragma(/*@non_null*/ ModelConstructorDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitModelTypePragma(/*@non_null*/ ModelTypePragma x, Object o);

  public abstract /*@non_null*/ Object visitModelMethodDeclPragma(/*@non_null*/ ModelMethodDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitGhostDeclPragma(/*@non_null*/ GhostDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitStillDeferredDeclPragma(/*@non_null*/ StillDeferredDeclPragma x, Object o);

  public abstract /*@non_null*/ Object visitSimpleStmtPragma(/*@non_null*/ SimpleStmtPragma x, Object o);

  public abstract /*@non_null*/ Object visitIdentifierModifierPragma(/*@non_null*/ IdentifierModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitExprStmtPragma(/*@non_null*/ ExprStmtPragma x, Object o);

  public abstract /*@non_null*/ Object visitSetStmtPragma(/*@non_null*/ SetStmtPragma x, Object o);

  public abstract /*@non_null*/ Object visitSkolemConstantPragma(/*@non_null*/ SkolemConstantPragma x, Object o);

  public abstract /*@non_null*/ Object visitModelProgamModifierPragma(/*@non_null*/ ModelProgamModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitNestedModifierPragma(/*@non_null*/ NestedModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitParsedSpecs(/*@non_null*/ ParsedSpecs x, Object o);

  public abstract /*@non_null*/ Object visitSimpleModifierPragma(/*@non_null*/ SimpleModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitExprModifierPragma(/*@non_null*/ ExprModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitModifiesGroupPragma(/*@non_null*/ ModifiesGroupPragma x, Object o);

  public abstract /*@non_null*/ Object visitCondExprModifierPragma(/*@non_null*/ CondExprModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitMapsExprModifierPragma(/*@non_null*/ MapsExprModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitReachModifierPragma(/*@non_null*/ ReachModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitVarDeclModifierPragma(/*@non_null*/ VarDeclModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitVarExprModifierPragma(/*@non_null*/ VarExprModifierPragma x, Object o);

  public abstract /*@non_null*/ Object visitNowarnPragma(/*@non_null*/ NowarnPragma x, Object o);

  public abstract /*@non_null*/ Object visitImportPragma(/*@non_null*/ ImportPragma x, Object o);

  public abstract /*@non_null*/ Object visitRefinePragma(/*@non_null*/ RefinePragma x, Object o);

  public abstract /*@non_null*/ Object visitSpec(/*@non_null*/ Spec x, Object o);

  public abstract /*@non_null*/ Object visitCondition(/*@non_null*/ Condition x, Object o);

  public abstract /*@non_null*/ Object visitDefPred(/*@non_null*/ DefPred x, Object o);

}
