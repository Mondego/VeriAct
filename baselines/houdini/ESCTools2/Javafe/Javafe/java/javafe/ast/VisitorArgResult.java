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

public abstract class VisitorArgResult {
  public abstract /*@non_null*/ Object visitASTNode(/*@non_null*/ ASTNode x, /*@nullable*/Object o);

  public /*@non_null*/ Object visitCompilationUnit(/*@non_null*/ CompilationUnit x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitImportDecl(/*@non_null*/ ImportDecl x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitSingleTypeImportDecl(/*@non_null*/ SingleTypeImportDecl x, /*@nullable*/Object o) {
    return visitImportDecl(x, o);
  }

  public /*@non_null*/ Object visitOnDemandImportDecl(/*@non_null*/ OnDemandImportDecl x, /*@nullable*/Object o) {
    return visitImportDecl(x, o);
  }

  public /*@non_null*/ Object visitTypeDecl(/*@non_null*/ TypeDecl x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitClassDecl(/*@non_null*/ ClassDecl x, /*@nullable*/Object o) {
    return visitTypeDecl(x, o);
  }

  public /*@non_null*/ Object visitInterfaceDecl(/*@non_null*/ InterfaceDecl x, /*@nullable*/Object o) {
    return visitTypeDecl(x, o);
  }

  public /*@non_null*/ Object visitRoutineDecl(/*@non_null*/ RoutineDecl x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitConstructorDecl(/*@non_null*/ ConstructorDecl x, /*@nullable*/Object o) {
    return visitRoutineDecl(x, o);
  }

  public /*@non_null*/ Object visitMethodDecl(/*@non_null*/ MethodDecl x, /*@nullable*/Object o) {
    return visitRoutineDecl(x, o);
  }

  public /*@non_null*/ Object visitInitBlock(/*@non_null*/ InitBlock x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitTypeDeclElemPragma(/*@non_null*/ TypeDeclElemPragma x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitGenericVarDecl(/*@non_null*/ GenericVarDecl x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitLocalVarDecl(/*@non_null*/ LocalVarDecl x, /*@nullable*/Object o) {
    return visitGenericVarDecl(x, o);
  }

  public /*@non_null*/ Object visitFieldDecl(/*@non_null*/ FieldDecl x, /*@nullable*/Object o) {
    return visitGenericVarDecl(x, o);
  }

  public /*@non_null*/ Object visitFormalParaDecl(/*@non_null*/ FormalParaDecl x, /*@nullable*/Object o) {
    return visitGenericVarDecl(x, o);
  }

  public /*@non_null*/ Object visitStmt(/*@non_null*/ Stmt x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitGenericBlockStmt(/*@non_null*/ GenericBlockStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitBlockStmt(/*@non_null*/ BlockStmt x, /*@nullable*/Object o) {
    return visitGenericBlockStmt(x, o);
  }

  public /*@non_null*/ Object visitSwitchStmt(/*@non_null*/ SwitchStmt x, /*@nullable*/Object o) {
    return visitGenericBlockStmt(x, o);
  }

  public /*@non_null*/ Object visitAssertStmt(/*@non_null*/ AssertStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitVarDeclStmt(/*@non_null*/ VarDeclStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitClassDeclStmt(/*@non_null*/ ClassDeclStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitWhileStmt(/*@non_null*/ WhileStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitDoStmt(/*@non_null*/ DoStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitSynchronizeStmt(/*@non_null*/ SynchronizeStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitEvalStmt(/*@non_null*/ EvalStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitReturnStmt(/*@non_null*/ ReturnStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitThrowStmt(/*@non_null*/ ThrowStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitBranchStmt(/*@non_null*/ BranchStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitBreakStmt(/*@non_null*/ BreakStmt x, /*@nullable*/Object o) {
    return visitBranchStmt(x, o);
  }

  public /*@non_null*/ Object visitContinueStmt(/*@non_null*/ ContinueStmt x, /*@nullable*/Object o) {
    return visitBranchStmt(x, o);
  }

  public /*@non_null*/ Object visitLabelStmt(/*@non_null*/ LabelStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitIfStmt(/*@non_null*/ IfStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitForStmt(/*@non_null*/ ForStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitSkipStmt(/*@non_null*/ SkipStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitSwitchLabel(/*@non_null*/ SwitchLabel x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitTryFinallyStmt(/*@non_null*/ TryFinallyStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitTryCatchStmt(/*@non_null*/ TryCatchStmt x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitStmtPragma(/*@non_null*/ StmtPragma x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitConstructorInvocation(/*@non_null*/ ConstructorInvocation x, /*@nullable*/Object o) {
    return visitStmt(x, o);
  }

  public /*@non_null*/ Object visitCatchClause(/*@non_null*/ CatchClause x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitVarInit(/*@non_null*/ VarInit x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitArrayInit(/*@non_null*/ ArrayInit x, /*@nullable*/Object o) {
    return visitVarInit(x, o);
  }

  public /*@non_null*/ Object visitExpr(/*@non_null*/ Expr x, /*@nullable*/Object o) {
    return visitVarInit(x, o);
  }

  public /*@non_null*/ Object visitThisExpr(/*@non_null*/ ThisExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitLiteralExpr(/*@non_null*/ LiteralExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitArrayRefExpr(/*@non_null*/ ArrayRefExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitNewInstanceExpr(/*@non_null*/ NewInstanceExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitNewArrayExpr(/*@non_null*/ NewArrayExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitCondExpr(/*@non_null*/ CondExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitInstanceOfExpr(/*@non_null*/ InstanceOfExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitCastExpr(/*@non_null*/ CastExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitBinaryExpr(/*@non_null*/ BinaryExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitUnaryExpr(/*@non_null*/ UnaryExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitParenExpr(/*@non_null*/ ParenExpr x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitAmbiguousVariableAccess(/*@non_null*/ AmbiguousVariableAccess x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitVariableAccess(/*@non_null*/ VariableAccess x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitFieldAccess(/*@non_null*/ FieldAccess x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitAmbiguousMethodInvocation(/*@non_null*/ AmbiguousMethodInvocation x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitMethodInvocation(/*@non_null*/ MethodInvocation x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitClassLiteral(/*@non_null*/ ClassLiteral x, /*@nullable*/Object o) {
    return visitExpr(x, o);
  }

  public /*@non_null*/ Object visitObjectDesignator(/*@non_null*/ ObjectDesignator x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitExprObjectDesignator(/*@non_null*/ ExprObjectDesignator x, /*@nullable*/Object o) {
    return visitObjectDesignator(x, o);
  }

  public /*@non_null*/ Object visitTypeObjectDesignator(/*@non_null*/ TypeObjectDesignator x, /*@nullable*/Object o) {
    return visitObjectDesignator(x, o);
  }

  public /*@non_null*/ Object visitSuperObjectDesignator(/*@non_null*/ SuperObjectDesignator x, /*@nullable*/Object o) {
    return visitObjectDesignator(x, o);
  }

  public /*@non_null*/ Object visitType(/*@non_null*/ Type x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitErrorType(/*@non_null*/ ErrorType x, /*@nullable*/Object o) {
    return visitType(x, o);
  }

  public /*@non_null*/ Object visitPrimitiveType(/*@non_null*/ PrimitiveType x, /*@nullable*/Object o) {
    return visitType(x, o);
  }

  public /*@non_null*/ Object visitJavafePrimitiveType(/*@non_null*/ JavafePrimitiveType x, /*@nullable*/Object o) {
    return visitPrimitiveType(x, o);
  }

  public /*@non_null*/ Object visitTypeName(/*@non_null*/ TypeName x, /*@nullable*/Object o) {
    return visitType(x, o);
  }

  public /*@non_null*/ Object visitArrayType(/*@non_null*/ ArrayType x, /*@nullable*/Object o) {
    return visitType(x, o);
  }

  public /*@non_null*/ Object visitName(/*@non_null*/ Name x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitSimpleName(/*@non_null*/ SimpleName x, /*@nullable*/Object o) {
    return visitName(x, o);
  }

  public /*@non_null*/ Object visitCompoundName(/*@non_null*/ CompoundName x, /*@nullable*/Object o) {
    return visitName(x, o);
  }

  public /*@non_null*/ Object visitModifierPragma(/*@non_null*/ ModifierPragma x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitLexicalPragma(/*@non_null*/ LexicalPragma x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

  public /*@non_null*/ Object visitTypeModifierPragma(/*@non_null*/ TypeModifierPragma x, /*@nullable*/Object o) {
    return visitASTNode(x, o);
  }

}
