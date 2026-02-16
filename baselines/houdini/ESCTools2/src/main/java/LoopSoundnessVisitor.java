/**
 * $Id: LoopSoundnessVisitor.java,v 1.2 2007/07/23 16:17:26 dcochran Exp $
 *
 * @title "Visitor for detection or unsoundness in loops"
 * @description "Walks through an AST and finds any cases where the right
 * shift operator is used. It then adds a warning to ErrorSet about this"
 */

package escjava.soundness;

import escjava.ast.ASTVisitor;
import escjava.ast.AnOverview;
import escjava.ast.ArrayRangeRefExpr;
import escjava.ast.CondExprModifierPragma;
import escjava.ast.Condition;
import escjava.ast.DecreasesInfo;
import escjava.ast.DefPred;
import escjava.ast.DefPredApplExpr;
import escjava.ast.DefPredLetExpr;
import escjava.ast.DependsPragma;
import escjava.ast.EscPrimitiveType;
import escjava.ast.EverythingExpr;
import escjava.ast.ExprDeclPragma;
import escjava.ast.ExprModifierPragma;
import escjava.ast.ExprStmtPragma;
import escjava.ast.GCExpr;
import escjava.ast.GhostDeclPragma;
import escjava.ast.GuardExpr;
import escjava.ast.GuardedCmd;
import escjava.ast.IdExprDeclPragma;
import escjava.ast.IdentifierModifierPragma;
import escjava.ast.ImportPragma;
import escjava.ast.LockSetExpr;
import escjava.ast.MapsExprModifierPragma;
import escjava.ast.ModelConstructorDeclPragma;
import escjava.ast.ModelDeclPragma;
import escjava.ast.ModelMethodDeclPragma;
import escjava.ast.ModelProgamModifierPragma;
import escjava.ast.ModelTypePragma;
import escjava.ast.ModifiesGroupPragma;
import escjava.ast.NamedExprDeclPragma;
import escjava.ast.NestedModifierPragma;
import escjava.ast.NotModifiedExpr;
import escjava.ast.NotSpecifiedExpr;
import escjava.ast.NothingExpr;
import escjava.ast.NowarnPragma;
import escjava.ast.ParsedSpecs;
import escjava.ast.ReachModifierPragma;
import escjava.ast.RefinePragma;
import escjava.ast.ResExpr;
import escjava.ast.SetCompExpr;
import escjava.ast.SetStmtPragma;
import escjava.ast.SimpleModifierPragma;
import escjava.ast.SimpleStmtPragma;
import escjava.ast.SkolemConstantPragma;
import escjava.ast.Spec;
import escjava.ast.StillDeferredDeclPragma;
import escjava.ast.TagConstants;
import escjava.ast.VarDeclModifierPragma;
import escjava.ast.VarExprModifierPragma;
import escjava.ast.WildRefExpr;
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
import javafe.ast.ClassDecl;
import javafe.ast.ClassDeclStmt;
import javafe.ast.ClassLiteral;
import javafe.ast.CompilationUnit;
import javafe.ast.CompoundName;
import javafe.ast.CondExpr;
import javafe.ast.ConstructorDecl;
import javafe.ast.ConstructorInvocation;
import javafe.ast.ContinueStmt;
import javafe.ast.DoStmt;
import javafe.ast.ErrorType;
import javafe.ast.EvalStmt;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.ForStmt;
import javafe.ast.FormalParaDecl;
import javafe.ast.GenericBlockStmt;
import javafe.ast.GenericVarDecl;
import javafe.ast.IfStmt;
import javafe.ast.ImportDecl;
import javafe.ast.InitBlock;
import javafe.ast.InstanceOfExpr;
import javafe.ast.InterfaceDecl;
import javafe.ast.LabelStmt;
import javafe.ast.LexicalPragma;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.MethodDecl;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.Name;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.OnDemandImportDecl;
import javafe.ast.ParenExpr;
import javafe.ast.PrimitiveType;
import javafe.ast.ReturnStmt;
import javafe.ast.RoutineDecl;
import javafe.ast.SimpleName;
import javafe.ast.SingleTypeImportDecl;
import javafe.ast.SkipStmt;
import javafe.ast.Stmt;
import javafe.ast.StmtPragma;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.SwitchLabel;
import javafe.ast.SwitchStmt;
import javafe.ast.SynchronizeStmt;
import javafe.ast.ThisExpr;
import javafe.ast.ThrowStmt;
import javafe.ast.TryCatchStmt;
import javafe.ast.TryFinallyStmt;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.TypeModifierPragma;
import javafe.ast.TypeName;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.UnaryExpr;
import javafe.ast.VarDeclStmt;
import javafe.ast.VarInit;
import javafe.ast.VariableAccess;
import javafe.ast.WhileStmt;
import javafe.util.ErrorSet;

public class LoopSoundnessVisitor extends ASTVisitor {

  public static final String LOOP_UNSOUNDNESS_WARNING = "The semantics of loops without loop invariants are unsound.";
/**
   * Indicates if a loop invariant exists
   */
  protected boolean loopInvariant = false;


public void visitCompilationUnit(/*@ non_null */ CompilationUnit x) {
    visitASTNode(x);
  }

  public void visitImportDecl(/*@ non_null */ ImportDecl x) {
    visitASTNode(x);
  }

  public void visitSingleTypeImportDecl(/*@ non_null */ SingleTypeImportDecl x) {
    visitImportDecl(x);
  }

  public void visitOnDemandImportDecl(/*@ non_null */ OnDemandImportDecl x) {
    visitImportDecl(x);
  }

  public void visitTypeDecl(/*@ non_null */ TypeDecl x) {
    visitASTNode(x);
  }

  public void visitClassDecl(/*@ non_null */ ClassDecl x) {
    visitTypeDecl(x);
  }

  public void visitInterfaceDecl(/*@ non_null */ InterfaceDecl x) {
    visitTypeDecl(x);
  }

  public void visitRoutineDecl(/*@ non_null */ RoutineDecl x) {
    visitASTNode(x);
  }

  public void visitConstructorDecl(/*@ non_null */ ConstructorDecl x) {
    visitRoutineDecl(x);
  }

  public void visitMethodDecl(/*@ non_null */ MethodDecl x) {
    visitRoutineDecl(x);
  }

  public void visitInitBlock(/*@ non_null */ InitBlock x) {
    visitASTNode(x);
  }

  public void visitTypeDeclElemPragma(/*@ non_null */ TypeDeclElemPragma x) {    
    visitASTNode(x);
  }

  public void visitGenericVarDecl(/*@ non_null */ GenericVarDecl x) {
    visitASTNode(x);
  }

  public void visitLocalVarDecl(/*@ non_null */ LocalVarDecl x) {
    visitGenericVarDecl(x);
  }

  public void visitFieldDecl(/*@ non_null */ FieldDecl x) {
    visitGenericVarDecl(x);
  }

  public void visitFormalParaDecl(/*@ non_null */ FormalParaDecl x) {
    visitGenericVarDecl(x);
  }

  public void visitStmt(/*@ non_null */ Stmt x) {
    visitASTNode(x);
  }

  public void visitGenericBlockStmt(/*@ non_null */ GenericBlockStmt x) {
    visitStmt(x);
  }

  public void visitBlockStmt(/*@ non_null */ BlockStmt x) {
    visitGenericBlockStmt(x);
  }

  public void visitSwitchStmt(/*@ non_null */ SwitchStmt x) {
    visitGenericBlockStmt(x);
  }

  public void visitAssertStmt(/*@ non_null */ AssertStmt x) {
    visitStmt(x);
  }

  public void visitVarDeclStmt(/*@ non_null */ VarDeclStmt x) {
    visitStmt(x);
  }

  public void visitClassDeclStmt(/*@ non_null */ ClassDeclStmt x) {
    visitStmt(x);
  }

  public void visitWhileStmt(/*@ non_null */ WhileStmt x) {
    visitStmt(x);
  }

  public void visitDoStmt(/*@ non_null */ DoStmt x) {
    visitStmt(x);
  }

  public void visitSynchronizeStmt(/*@ non_null */ SynchronizeStmt x) {
    visitStmt(x);
  }

  public void visitEvalStmt(/*@ non_null */ EvalStmt x) {
    visitStmt(x);
  }

  public void visitReturnStmt(/*@ non_null */ ReturnStmt x) {
    visitStmt(x);
  }

  public void visitThrowStmt(/*@ non_null */ ThrowStmt x) {
    visitStmt(x);
  }

  public void visitBranchStmt(/*@ non_null */ BranchStmt x) {
    visitStmt(x);
  }

  public void visitBreakStmt(/*@ non_null */ BreakStmt x) {
    visitBranchStmt(x);
  }

  public void visitContinueStmt(/*@ non_null */ ContinueStmt x) {
    visitBranchStmt(x);
  }

  public void visitLabelStmt(/*@ non_null */ LabelStmt x) {
    visitStmt(x);
  }

  public void visitIfStmt(/*@ non_null */ IfStmt x) {
    visitStmt(x);
  }

  public void visitForStmt(/*@ non_null */ ForStmt x) {
    
    if (!loopInvariant) {
      ErrorSet.warning(x.getStartLoc(), LOOP_UNSOUNDNESS_WARNING);
    }
    loopInvariant = false;
    
    visitStmt(x);
  }

  public void visitSkipStmt(/*@ non_null */ SkipStmt x) {
    visitStmt(x);
  }

  public void visitSwitchLabel(/*@ non_null */ SwitchLabel x) {
    visitStmt(x);
  }

  public void visitTryFinallyStmt(/*@ non_null */ TryFinallyStmt x) {
    visitStmt(x);
  }

  public void visitTryCatchStmt(/*@ non_null */ TryCatchStmt x) {
    visitStmt(x);
  }

  public void visitStmtPragma(/*@ non_null */ StmtPragma x) {
     
     if (x.getTag() == TagConstants.LOOP_INVARIANT) {
        loopInvariant = true;
     }
     
    visitStmt(x);
  }

  public void visitConstructorInvocation(/*@ non_null */ ConstructorInvocation x) {
    visitStmt(x);
  }

  public void visitCatchClause(/*@ non_null */ CatchClause x) {
    visitASTNode(x);
  }

  public void visitVarInit(/*@ non_null */ VarInit x) {
    visitASTNode(x);
  }

  public void visitArrayInit(/*@ non_null */ ArrayInit x) {
    visitVarInit(x);
  }

  public void visitExpr(/*@ non_null */ Expr x) {
    visitVarInit(x);
  }

  public void visitThisExpr(/*@ non_null */ ThisExpr x) {
    visitExpr(x);
  }

  public void visitLiteralExpr(/*@ non_null */ LiteralExpr x) {
    visitExpr(x);
  }

  public void visitArrayRefExpr(/*@ non_null */ ArrayRefExpr x) {
    visitExpr(x);
  }

  public void visitNewInstanceExpr(/*@ non_null */ NewInstanceExpr x) {
    visitExpr(x);
  }

  public void visitNewArrayExpr(/*@ non_null */ NewArrayExpr x) {
    visitExpr(x);
  }

  public void visitCondExpr(/*@ non_null */ CondExpr x) {
    visitExpr(x);
  }

  public void visitInstanceOfExpr(/*@ non_null */ InstanceOfExpr x) {
    visitExpr(x);
  }

  public void visitCastExpr(/*@ non_null */ CastExpr x) {
    visitExpr(x);
  }

  public void visitBinaryExpr(/*@ non_null */ BinaryExpr x) {
    visitExpr(x);
  }

  public void visitUnaryExpr(/*@ non_null */ UnaryExpr x) {
    visitExpr(x);
  }

  public void visitParenExpr(/*@ non_null */ ParenExpr x) {
    visitExpr(x);
  }

  public void visitAmbiguousVariableAccess(/*@ non_null */ AmbiguousVariableAccess x) {
    visitExpr(x);
  }

  public void visitVariableAccess(/*@ non_null */ VariableAccess x) {
    visitExpr(x);
  }

  public void visitFieldAccess(/*@ non_null */ FieldAccess x) {
    visitExpr(x);
  }

  public void visitAmbiguousMethodInvocation(/*@ non_null */ AmbiguousMethodInvocation x) {
    visitExpr(x);
  }

  public void visitMethodInvocation(/*@ non_null */ MethodInvocation x) {
    visitExpr(x);
  }

  public void visitClassLiteral(/*@ non_null */ ClassLiteral x) {
    visitExpr(x);
  }

  public void visitObjectDesignator(/*@ non_null */ ObjectDesignator x) {
    visitASTNode(x);
  }

  public void visitExprObjectDesignator(/*@ non_null */ ExprObjectDesignator x) {
    visitObjectDesignator(x);
  }

  public void visitTypeObjectDesignator(/*@ non_null */ TypeObjectDesignator x) {
    visitObjectDesignator(x);
  }

  public void visitSuperObjectDesignator(/*@ non_null */ SuperObjectDesignator x) {
    visitObjectDesignator(x);
  }

  public void visitType(/*@ non_null */ Type x) {
    visitASTNode(x);
  }

  public void visitErrorType(/*@ non_null */ ErrorType x) {
    visitType(x);
  }

  public void visitPrimitiveType(/*@ non_null */ PrimitiveType x) {
    visitType(x);
  }

  public void visitTypeName(/*@ non_null */ TypeName x) {
    visitType(x);
  }

  public void visitArrayType(/*@ non_null */ ArrayType x) {
    visitType(x);
  }

  public void visitName(/*@ non_null */ Name x) {
    visitASTNode(x);
  }

  public void visitSimpleName(/*@ non_null */ SimpleName x) {
    visitName(x);
  }

  public void visitCompoundName(/*@ non_null */ CompoundName x) {
    visitName(x);
  }

  public void visitModifierPragma(/*@ non_null */ ModifierPragma x) {
    visitASTNode(x);
  }

  public void visitLexicalPragma(/*@ non_null */ LexicalPragma x) {
    visitASTNode(x);
  }

  public void visitTypeModifierPragma(/*@ non_null */ TypeModifierPragma x) {
    visitASTNode(x);
  }
  
  public void visitAnOverview(/*@ non_null */ AnOverview x) {
     visitASTNode(x);
     
  }


  public void visitArrayRangeRefExpr(/*@ non_null */ ArrayRangeRefExpr x) {
     visitASTNode(x);
     
  }


  public void visitCondExprModifierPragma(/*@ non_null */ CondExprModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitCondition(/*@ non_null */ Condition x) {
     visitASTNode(x);
     
  }


  public void visitDecreasesInfo(/*@ non_null */ DecreasesInfo x) {
     visitASTNode(x);
     
  }


  public void visitDefPred(/*@ non_null */ DefPred x) {
     visitASTNode(x);
     
  }


  public void visitDefPredApplExpr(/*@ non_null */ DefPredApplExpr x) {
     visitASTNode(x);
     
  }


  public void visitDefPredLetExpr(/*@ non_null */ DefPredLetExpr x) {
     visitASTNode(x);
     
  }


  public void visitDependsPragma(/*@ non_null */ DependsPragma x) {
     visitASTNode(x);
     
  }


  public void visitEscPrimitiveType(/*@ non_null */ EscPrimitiveType x) {
     visitASTNode(x);
     
  }


  public void visitEverythingExpr(/*@ non_null */ EverythingExpr x) {
     visitASTNode(x);
     
  }


  public void visitExprDeclPragma(/*@ non_null */ ExprDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitExprModifierPragma(/*@ non_null */ ExprModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitExprStmtPragma(/*@ non_null */ ExprStmtPragma x) {
     visitASTNode(x);
     
  }


  public void visitGCExpr(/*@ non_null */ GCExpr x) {
     visitASTNode(x);
     
  }


  public void visitGhostDeclPragma(/*@ non_null */ GhostDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitGuardExpr(/*@ non_null */ GuardExpr x) {
     visitASTNode(x);
     
  }


  public void visitGuardedCmd(/*@ non_null */ GuardedCmd x) {
     visitASTNode(x);
     
  }


  public void visitIdExprDeclPragma(/*@ non_null */ IdExprDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitIdentifierModifierPragma(/*@ non_null */ IdentifierModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitImportPragma(/*@ non_null */ ImportPragma x) {
     visitASTNode(x);
     
  }


  public void visitLockSetExpr(/*@ non_null */ LockSetExpr x) {
     visitASTNode(x);
     
  }


  public void visitMapsExprModifierPragma(/*@ non_null */ MapsExprModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitModelConstructorDeclPragma(/*@ non_null */ ModelConstructorDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitModelDeclPragma(/*@ non_null */ ModelDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitModelMethodDeclPragma(/*@ non_null */ ModelMethodDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitModelProgamModifierPragma(/*@ non_null */ ModelProgamModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitModelTypePragma(/*@ non_null */ ModelTypePragma x) {
     visitASTNode(x);
     
  }


  public void visitModifiesGroupPragma(/*@ non_null */ ModifiesGroupPragma x) {
     visitASTNode(x);
     
  }


  public void visitNamedExprDeclPragma(/*@ non_null */ NamedExprDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitNestedModifierPragma(/*@ non_null */ NestedModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitNotModifiedExpr(/*@ non_null */ NotModifiedExpr x) {
     visitASTNode(x);
     
  }


  public void visitNotSpecifiedExpr(/*@ non_null */ NotSpecifiedExpr x) {
     visitASTNode(x);
     
  }


  public void visitNothingExpr(/*@ non_null */ NothingExpr x) {
     visitASTNode(x);
     
  }


  public void visitNowarnPragma(/*@ non_null */ NowarnPragma x) {
     visitASTNode(x);
     
  }


  public void visitParsedSpecs(/*@ non_null */ ParsedSpecs x) {
     visitASTNode(x);
     
  }


  public void visitReachModifierPragma(/*@ non_null */ ReachModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitRefinePragma(/*@ non_null */ RefinePragma x) {
     visitASTNode(x);
     
  }


  public void visitResExpr(/*@ non_null */ ResExpr x) {
     visitASTNode(x);
     
  }


  public void visitSetCompExpr(/*@ non_null */ SetCompExpr x) {
     visitASTNode(x);
     
  }


  public void visitSetStmtPragma(/*@ non_null */ SetStmtPragma x) {
     visitASTNode(x);
     
  }


  public void visitSimpleModifierPragma(/*@ non_null */ SimpleModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitSimpleStmtPragma(/*@ non_null */ SimpleStmtPragma x) {
     visitASTNode(x);
     
  }


  public void visitSpec(/*@ non_null */ Spec x) {
     visitASTNode(x);
     
  }


  public void visitStillDeferredDeclPragma(/*@ non_null */ StillDeferredDeclPragma x) {
     visitASTNode(x);
     
  }


  public void visitVarDeclModifierPragma(/*@ non_null */ VarDeclModifierPragma x) {
     visitASTNode(x);
     
  }


  public void visitWildRefExpr(/*@ non_null */ WildRefExpr x) {
     visitASTNode(x);
     
  }
  
  public void visitSkolemConstantPragma(/*@ non_null */ SkolemConstantPragma x) {
     visitASTNode(x);
     
  }


  public void visitVarExprModifierPragma(/*@ non_null */ VarExprModifierPragma x) {
     visitASTNode(x);
     
  }


}
