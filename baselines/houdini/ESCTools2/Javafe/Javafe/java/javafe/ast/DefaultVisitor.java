// $Id$
/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;

import javafe.util.Assert;
import javafe.util.Location;


public class DefaultVisitor extends Visitor {

  public void visitASTNode(/*@non_null*/ ASTNode x) {
    // Assert.fail("DefaultVisitor.visitASTNode"+x);
  }

  public void visitModifierPragma(/*@non_null*/ ModifierPragma x) {
  }

  public void visitTypeDeclElemPragma(/*@non_null*/ TypeDeclElemPragma x) {
  }

  public void visitStmtPragma(/*@non_null*/ StmtPragma x) {
  }

  public void visitTypeModifierPragma(/*@non_null*/ TypeModifierPragma x) {
  }

  public void visitLexicalPragma(/*@non_null*/ LexicalPragma x) {
  }


  public void visitCompilationUnit(/*@non_null*/ CompilationUnit x) {
    for(int i=0; i<x.imports.size(); i++)
      x.imports.elementAt(i).accept( this );
    for(int i=0; i<x.elems.size(); i++)
      x.elems.elementAt(i).accept( this );
    if( x.lexicalPragmas != null ) 
	for(int i=0; i<x.lexicalPragmas.size(); i++)
	    x.lexicalPragmas.elementAt(i).accept( this );
  }

  public void visitTypeDecl(/*@non_null*/ TypeDecl x) {
    for(int i=0; i<x.superInterfaces.size(); i++)
      x.superInterfaces.elementAt(i).accept( this );
    for(int i=0; i<x.elems.size(); i++)
      x.elems.elementAt(i).accept( this );
    if( x.pmodifiers != null ) 
	for(int i=0; i<x.pmodifiers.size(); i++)
	    x.pmodifiers.elementAt(i).accept( this );
    if( x.tmodifiers != null ) 
	for(int i=0; i<x.tmodifiers.size(); i++)
	    x.tmodifiers.elementAt(i).accept( this );
  }

  public void visitClassDecl(/*@non_null*/ ClassDecl x) {
    if( x.superClass != null ) x.superClass.accept( this );
    visitTypeDecl(x);
  }

  public void visitInterfaceDecl(/*@non_null*/ InterfaceDecl x) {
    visitTypeDecl(x);
  }

  public void visitRoutineDecl(/*@non_null*/ RoutineDecl x) {
    for(int i=0; i<x.args.size(); i++)
      x.args.elementAt(i).accept( this );
    for(int i=0; i<x.raises.size(); i++)
      x.raises.elementAt(i).accept( this );
    if( x.pmodifiers != null ) 
	for(int i=0; i<x.pmodifiers.size(); i++)
	    x.pmodifiers.elementAt(i).accept( this );
    if( x.body != null ) x.body.accept( this );

  }

  public void visitConstructorDecl(/*@non_null*/ ConstructorDecl x) {
    visitRoutineDecl(x);
    if( x.tmodifiers != null ) 
	for(int i=0; i<x.tmodifiers.size(); i++)
	    x.tmodifiers.elementAt(i).accept( this );
  }

  public void visitMethodDecl(/*@non_null*/ MethodDecl x) {
    visitRoutineDecl(x);
    x.returnType.accept( this );
  }

  public void visitInitBlock(/*@non_null*/ InitBlock x) {
    x.block.accept( this );
    if( x.pmodifiers != null ) 
	for(int i=0; i<x.pmodifiers.size(); i++)
	    x.pmodifiers.elementAt(i).accept( this );
  }

  public void visitGenericVarDecl(/*@non_null*/ GenericVarDecl x) {
    if( x.pmodifiers != null ) 
	for(int i=0; i<x.pmodifiers.size(); i++)
	    x.pmodifiers.elementAt(i).accept( this );
    x.type.accept( this );
  }

  public void visitLocalVarDecl(/*@non_null*/ LocalVarDecl x) {
    visitGenericVarDecl(x);
    if( x.init != null ) x.init.accept( this );
  }

  public void visitFieldDecl(/*@non_null*/ FieldDecl x) {
    visitGenericVarDecl(x);
    if( x.init != null ) x.init.accept( this );
  }

  public void visitFormalParaDecl(/*@non_null*/ FormalParaDecl x) {
    visitGenericVarDecl(x);
  }

  // ------------------------------

  public void visitGenericBlockStmt(/*@non_null*/ GenericBlockStmt x) {
    for(int i=0; i<x.stmts.size(); i++)
      x.stmts.elementAt(i).accept( this );
  }

  public void visitBlockStmt(/*@non_null*/ BlockStmt x) {
    visitGenericBlockStmt(x);
  }

  public void visitSwitchStmt(/*@non_null*/ SwitchStmt x) {
    visitGenericBlockStmt(x);
    x.expr.accept( this );
  }

  public void visitClassDeclStmt(/*@non_null*/ ClassDeclStmt x) {
    x.decl.accept( this );
  }

  public void visitVarDeclStmt(/*@non_null*/ VarDeclStmt x) {
    x.decl.accept( this );
  }

  public void visitWhileStmt(/*@non_null*/ WhileStmt x) {
    x.expr.accept( this );
    x.stmt.accept( this );
  }

  public void visitDoStmt(/*@non_null*/ DoStmt x) {
    x.expr.accept( this );
    x.stmt.accept( this );
  }

  public void visitSynchronizeStmt(/*@non_null*/ SynchronizeStmt x) {
    x.expr.accept( this );
    x.stmt.accept( this );
  }

  public void visitEvalStmt(/*@non_null*/ EvalStmt x) {
    x.expr.accept( this );
  }

  public void visitReturnStmt(/*@non_null*/ ReturnStmt x) {
    if( x.expr != null ) x.expr.accept( this );
  }

  public void visitThrowStmt(/*@non_null*/ ThrowStmt x) {
    x.expr.accept( this );
  }

  public void visitBranchStmt(/*@non_null*/ BranchStmt x) {
  }

  public void visitBreakStmt(/*@non_null*/ BreakStmt x) {
  }

  public void visitContinueStmt(/*@non_null*/ ContinueStmt x) {
  }

  public void visitLabelStmt(/*@non_null*/ LabelStmt x) {
    x.stmt.accept( this );
  }

  public void visitIfStmt(/*@non_null*/ IfStmt x) {
    x.expr.accept( this );
    x.thn.accept( this );
    x.els.accept( this );
  }

  public void visitForStmt(/*@non_null*/ ForStmt x) {
    for(int i=0; i<x.forInit.size(); i++)
      x.forInit.elementAt(i).accept( this );
    x.test.accept( this );
    for(int i=0; i<x.forUpdate.size(); i++)
      x.forUpdate.elementAt(i).accept( this );
    x.body.accept( this );
  }

  public void visitSkipStmt(/*@non_null*/ SkipStmt x) {
  }

  public void visitSwitchLabel(/*@non_null*/ SwitchLabel x) {
    if( x.expr != null ) x.expr.accept( this );
  }

  public void visitTryFinallyStmt(/*@non_null*/ TryFinallyStmt x) {
    x.tryClause.accept( this );
    x.finallyClause.accept( this );
  }

  public void visitTryCatchStmt(/*@non_null*/ TryCatchStmt x) {
    x.tryClause.accept( this );
    for(int i=0; i<x.catchClauses.size(); i++)
      x.catchClauses.elementAt(i).accept( this );
  }

  public void visitConstructorInvocation(/*@non_null*/ ConstructorInvocation x) {
    for(int i=0; i<x.args.size(); i++)
      x.args.elementAt(i).accept( this );
  }

  public void visitCatchClause(/*@non_null*/ CatchClause x) {
    x.arg.accept( this );
    x.body.accept( this );
  }

  public void visitArrayInit(/*@non_null*/ ArrayInit x) {
    for(int i=0; i<x.elems.size(); i++)
      x.elems.elementAt(i).accept( this );
  }

  public void visitExpr(/*@non_null*/ Expr x) {
  }

  public void visitThisExpr(/*@non_null*/ ThisExpr x) {
    if (x.classPrefix != null)
	x.classPrefix.accept( this );
    visitExpr(x);
  }

  public void visitLiteralExpr(/*@non_null*/ LiteralExpr x) {
    visitExpr(x);
  }

  public void visitArrayRefExpr(/*@non_null*/ ArrayRefExpr x) {
    x.array.accept( this );
    x.index.accept( this );
    visitExpr(x);
  }

  public void visitNewInstanceExpr(/*@non_null*/ NewInstanceExpr x) {
    x.type.accept( this );
    for(int i=0; i<x.args.size(); i++)
      x.args.elementAt(i).accept( this );
    if (x.anonDecl != null) x.anonDecl.accept( this );
    visitExpr(x);
  }

  public void visitNewArrayExpr(/*@non_null*/ NewArrayExpr x) {
    x.type.accept( this );
    for(int i=0; i<x.dims.size(); i++)
      x.dims.elementAt(i).accept( this );
    if (x.init != null) x.init.accept( this );
    visitExpr(x);
  }

  public void visitCondExpr(/*@non_null*/ CondExpr x) {
    x.test.accept( this );
    x.thn.accept( this );
    x.els.accept( this );
    visitExpr(x);
  }

  public void visitInstanceOfExpr(/*@non_null*/ InstanceOfExpr x) {
    x.expr.accept( this );
    x.type.accept( this );
    visitExpr(x);
  }

  public void visitCastExpr(/*@non_null*/ CastExpr x) {
    x.expr.accept( this );
    x.type.accept( this );
    visitExpr(x);
  }

  public void visitBinaryExpr(/*@non_null*/ BinaryExpr x) {
    x.left.accept( this );
    x.right.accept( this );
    visitExpr(x);
  }

  public void visitUnaryExpr(/*@non_null*/ UnaryExpr x) {
    x.expr.accept( this );
    visitExpr(x);
  }

  public void visitParenExpr(/*@non_null*/ ParenExpr x) {
    x.expr.accept( this );
    visitExpr(x);
  }

  public void visitAmbiguousVariableAccess(/*@non_null*/ AmbiguousVariableAccess x) {
    visitExpr(x);
  }

  public void visitVariableAccess(/*@non_null*/ VariableAccess x) {
    visitExpr(x);
  }

  public void visitFieldAccess(/*@non_null*/ FieldAccess x) {
    x.od.accept( this );
    visitExpr(x);
  }

  public void visitAmbiguousMethodInvocation(/*@non_null*/ AmbiguousMethodInvocation x) {
    for(int i=0; i<x.args.size(); i++)
      x.args.elementAt(i).accept( this );
    visitExpr(x);
  }

  public void visitMethodInvocation(/*@non_null*/ MethodInvocation x) {
    x.od.accept( this );
    for(int i=0; i<x.args.size(); i++)
      x.args.elementAt(i).accept( this );
    visitExpr(x);
  }

  public void visitClassLiteral(/*@non_null*/ ClassLiteral x) {
    x.type.accept( this );
    visitExpr(x);
  }


  // ------------------------------

  public void visitExprObjectDesignator(/*@non_null*/ ExprObjectDesignator x) {
    x.expr.accept( this );
    visitObjectDesignator(x);
  }

  public void visitTypeObjectDesignator(/*@non_null*/ TypeObjectDesignator x) {
    x.type.accept( this );
    visitObjectDesignator(x);
  }

  public void visitSuperObjectDesignator(/*@non_null*/ SuperObjectDesignator x) {
    visitObjectDesignator(x);
  }

  // ------------------------------

  public void visitPrimitiveType(/*@non_null*/ PrimitiveType x) {
    visitType(x);
  }

  public void visitTypeName(/*@non_null*/ TypeName x) {
    visitType(x);
  }

  public void visitArrayType(/*@non_null*/ ArrayType x) {
    x.elemType.accept( this );
    visitType(x);
  }

  public void visitType(/*@non_null*/ Type x) {
  }

  // ------------------------------

  public void visitName(/*@non_null*/ Name x) {
  }
}
