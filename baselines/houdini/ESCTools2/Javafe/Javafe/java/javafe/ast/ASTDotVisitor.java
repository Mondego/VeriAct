/**
 * $Id$
 *
 * @title "AST DOT file visitor"
 * @description "Walks through an AST and outputs a dot graph format representation of that AST"
 * @
 */

package javafe.ast;

import java.io.ByteArrayOutputStream;
import java.io.FileWriter;
import java.io.IOException;

public class ASTDotVisitor extends Visitor {
  
  /**
   * our means of outputting the dot representation of the AST 
   */
  private FileWriter out = null;
  
  /**
   * a means of getting source text from the pretty printer
   */
  private ByteArrayOutputStream source_dump = null;
  
  /**
   * stores result of the last visit method call
   */
  private String visit_string = "";
  
  /**
   * pretty printer for extracting our source text
   */
  private StandardPrettyPrint printer = null;
  
  /**
   * Sets up the visit string for writing to a dot file with the given
   * source text, dotId and node type. This will also reset the buffer which
   * the pretty printer writes to.
   */
  private void setVisitString(String source_text, int dotId, String type) {
    // set the string
    visit_string = '"' + "" + dotId + " " + type + " " + source_text + '"';
    // reset the buffer
    source_dump.reset();
  }
  
  /**
   * recurses through the tree and writes the dotfile representation of it. if
   * parent is null this is the root of the tree. current is the node we are
   * parsing.
   */
  private void writeDot(ASTNode parent, ASTNode current) {
    // if this node is the root of the tree start the recursion otherwise make
    // the links between this node and its parent then do the recursive call
    // on the children
    if(parent != null) {
      // contains the id and type of the parent node
      String parent_representation;
      // contains the id and type of this node
      String current_representation;
      // call the visitor on parent and current and retrieve the string descriptions
      parent.accept(this);
      parent_representation = visit_string;
      current.accept(this);
      current_representation = visit_string;
      // make the link in the file
      try {
        out.write("  " + parent_representation + " -> " + current_representation + "\n");
      } catch (IOException exception) {
        System.out.println("failed to write to file");
      }
    }
    // do the recursive call on the children provided the child is an ASTNode
    for(int count = 0; count < current.childCount(); count++) {
      Object o = current.childAt(count);
      if(o instanceof ASTNode) writeDot(current, (ASTNode) o);
    }
  }
  
  /**
   * constructor that takes in a compilation unit that represents a class
   * and initalizes a filewriter in the directory escjava is called from
   */
  public ASTDotVisitor() {
    // holds the value of this node so when we do recursive calls we can make
    // the correct edges in the graph
    //
    // try opening the file for now throw a message out if it fails
    try {
      out = new FileWriter("Test.dot");
    } catch(IOException exception) {
      System.out.println("could not open file");
    }
    // init source_dump and printer
    source_dump = new ByteArrayOutputStream();
    printer = new StandardPrettyPrint();
  }
  
  /**
   * visits an ast node and outputs it in dot file format
   */
  public void visitASTNode(/*@ non_null */ ASTNode x) {
    // write the file
    try {
      // write the start of the dot file
      out.write("digraph Test {\n");
      // call the recursive method on x
      writeDot(null, x);
      // write the end of the dot file
      out.write("}\n");
      out.close();
    } catch(IOException exception) {
      System.out.println("failed to write to file");
    }
  }
  
  public void visitCompilationUnit(/*@ non_null */ CompilationUnit x) {
    setVisitString("", x.dotId, "CompilationUnit(todo)");
  }
  
  public void visitImportDecl(/*@ non_null */ ImportDecl x) {
    setVisitString("", x.dotId, "ImportDecl(have not seen yet)");
  }
  
  public void visitSingleTypeImportDecl(/*@ non_null */ SingleTypeImportDecl x) {
    setVisitString("", x.dotId, "SingleTypeImportDecl(todo)");
  }
  
  public void visitOnDemandImportDecl(/*@ non_null */ OnDemandImportDecl x) {
    setVisitString("", x.dotId, "OnDemandImportDecl(todo)");
  }

  public void visitTypeDecl(/*@ non_null */ TypeDecl x) {
    // get the source text
    printer.print(source_dump, 0, x);
  }

  public void visitClassDecl(/*@ non_null */ ClassDecl x) {
    visitTypeDecl(x);
    // we need to grab everything before the first { charcter and remove
    // all the newlines
    String source = source_dump.toString();
    char[] source_array = source.substring(0, source.indexOf('{')).toCharArray();
    for(int count = 0; count < source_array.length; count++) {
      if(source_array[count] == '\n') source_array[count] = ' ';
    }
    source = new String(source_array);
    setVisitString(source, x.dotId, "ClassDecl");
  }

  public void visitInterfaceDecl(/*@ non_null */ InterfaceDecl x) {
    visitTypeDecl(x);
    // we have to be pretty careful here as we can have interfaces that extend
    // other interfaces so we need to get everything before the first {
    // charcter. we also need to replace the newline charcters as well
    String source = source_dump.toString();
    char[] source_array = source.substring(0, source.indexOf('{')).toCharArray();
    for(int count = 0; count < source_array.length; count++) {
      if(source_array[count] == '\n') source_array[count] = ' ';
    }
    source = new String(source_array);
    setVisitString(source, x.dotId, "InterfaceDecl");
  }

  public void visitRoutineDecl(/*@ non_null */ RoutineDecl x) {
  }

  public void visitConstructorDecl(/*@ non_null */ ConstructorDecl x) {
    // grab the source text
    printer.print(source_dump, 0, x, x.id(), false);
    // get everything before the first newline
    String source = source_dump.toString().substring(0,source_dump.toString().indexOf('\n'));
    setVisitString(source, x.dotId, "ConstructorDecl");
  }

  public void visitMethodDecl(/*@ non_null */ MethodDecl x) {
    // grab the source text
    printer.print(source_dump, 0, x, x.id(), false);
    // grab everything before the first newline
    String source = source_dump.toString().substring(0,source_dump.toString().indexOf('\n'));
    setVisitString(source, x.dotId, "MethodDecl");
  }

  public void visitInitBlock(/*@ non_null */ InitBlock x) {
    setVisitString("", x.dotId, "InitBlock(have not seen yet)");
  }

  public void visitTypeDeclElemPragma(/*@ non_null */ TypeDeclElemPragma x) {
    setVisitString("", x.dotId, "TypeDeclElemPragma(have not seen yet)");
  }

  public void visitGenericVarDecl(/*@ non_null */ GenericVarDecl x) {
    // get the source text
    printer.print(source_dump, x);
  }

  public void visitLocalVarDecl(/*@ non_null */ LocalVarDecl x) {
    visitGenericVarDecl(x);
    setVisitString(source_dump.toString(), x.dotId, "LocalVarDecl");
  }

  public void visitFieldDecl(/*@ non_null */ FieldDecl x) {
    visitGenericVarDecl(x);
    setVisitString(source_dump.toString(), x.dotId, "FieldDecl");
  }

  public void visitFormalParaDecl(/*@ non_null */ FormalParaDecl x) {
    visitGenericVarDecl(x);
    setVisitString(source_dump.toString(), x.dotId, "FormalParaDecl");
  }

  public void visitStmt(/*@ non_null */ Stmt x) {
    // grab the source text
    printer.print(source_dump, 0, x);
  }

  public void visitGenericBlockStmt(/*@ non_null */ GenericBlockStmt x) {
    visitStmt(x);
  }

  public void visitBlockStmt(/*@ non_null */ BlockStmt x) {
    visitGenericBlockStmt(x);
    setVisitString("", x.dotId, "BlockStmt");
  }

  public void visitSwitchStmt(/*@ non_null */ SwitchStmt x) {
    visitStmt(x);
    // grab everything the first occurence of the ) charcter searching
    // backwards from the first newline
    String source = source_dump.toString();
    source = source.substring(0, source.lastIndexOf(')', source.indexOf('\n')) + 1);
    setVisitString(source, x.dotId, "SwitchStmt");
  }

  public void visitAssertStmt(/*@ non_null */ AssertStmt x) {
    setVisitString("", x.dotId, "AssertStmt(have not seen yet)");
  }

  public void visitVarDeclStmt(/*@ non_null */ VarDeclStmt x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "VarDeclStmt");
  }

  public void visitClassDeclStmt(/*@ non_null */ ClassDeclStmt x) {
    setVisitString("", x.dotId, "ClassDeclStmt(have not seen yet)");
  }

  public void visitWhileStmt(/*@ non_null */ WhileStmt x) {
    visitStmt(x);
    // grab everything from the start of the while statment to the last )
    // charcter from the first newline
    String source = source_dump.toString();
    source = source.substring(0, source.lastIndexOf(')', source.indexOf('\n')) + 1);
    setVisitString(source, x.dotId, "WhileStmt");
  }

  public void visitDoStmt(/*@ non_null */ DoStmt x) {
    visitStmt(x);
    setVisitString("", x.dotId, "DoStmt");
  }

  public void visitSynchronizeStmt(/*@ non_null */ SynchronizeStmt x) {
    setVisitString("", x.dotId, "SynchronizeStmt(have not seen yet)");
  }

  public void visitEvalStmt(/*@ non_null */ EvalStmt x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "EvalStmt");
  }

  public void visitReturnStmt(/*@ non_null */ ReturnStmt x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "ReturnStmt");
  }

  public void visitThrowStmt(/*@ non_null */ ThrowStmt x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "ThrowStmt");
  }

  public void visitBranchStmt(/*@ non_null */ BranchStmt x) {
    setVisitString("", x.dotId, "BranchStmt(have not seen yet)");
  }

  public void visitBreakStmt(/*@ non_null */ BreakStmt x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "BreakStmt");
  }

  public void visitContinueStmt(/*@ non_null */ ContinueStmt x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "ContinueStmt");
  }

  public void visitLabelStmt(/*@ non_null */ LabelStmt x) {
    setVisitString("", x.dotId, "LabelStmt(have not seen yet)");
  }

  public void visitIfStmt(/*@ non_null */ IfStmt x) {
    setVisitString("", x.dotId, "IfStmt");
  }

  public void visitForStmt(/*@ non_null */ ForStmt x) {
    visitStmt(x);
    // get everything from the beginning until the last ) charcter is
    // encountered on the first line
    String source = source_dump.toString();
    source = source.substring(0, source.lastIndexOf(')', source.indexOf('\n')) + 1);
    setVisitString(source, x.dotId, "ForStmt");
  }

  public void visitSkipStmt(/*@ non_null */ SkipStmt x) {
    visitStmt(x);
    setVisitString("", x.dotId, "SkipStmt");
  }

  public void visitSwitchLabel(/*@ non_null */ SwitchLabel x) {
    // source string to append to the node label
    String source = null;
    visitStmt(x);
    // we must remove the new line from the label if there is one
    if(source_dump.toString().indexOf('\n') != -1) source = source_dump.toString().substring(0, source_dump.toString().indexOf('\n'));
    else source = source_dump.toString();
    setVisitString(source, x.dotId, "SwitchLabel");
  }

  public void visitTryFinallyStmt(/*@ non_null */ TryFinallyStmt x) {
    visitStmt(x.finallyClause);
    setVisitString("", x.dotId, "TryFinallyStmt");
  }

  public void visitTryCatchStmt(/*@ non_null */ TryCatchStmt x) {
    visitStmt(x);
    // grab the try statment before the first { charcter
    String source = source_dump.toString();
    source = source.substring(0, source.indexOf('{'));
    setVisitString(source, x.dotId, "TryCatchStmt");
  }

  public void visitStmtPragma(/*@ non_null */ StmtPragma x) {
    setVisitString("", x.dotId, "StmtPragma(have not seen yet)");
  }

  public void visitConstructorInvocation(/*@ non_null */ ConstructorInvocation x) {
    visitStmt(x);
    setVisitString(source_dump.toString(), x.dotId, "ConstructorInvocation");
  }

  public void visitCatchClause(/*@ non_null */ CatchClause x) {
    // we have to do some manual work here as the catch clause is not
    // really dealt with on its own in the pretty printer.
    printer.print(source_dump, x.arg);
    
    setVisitString("catch (" + source_dump.toString() + ")", x.dotId, "CatchClause");
  }

  public void visitVarInit(/*@ non_null */ VarInit x) {
    // get the source text
    printer.print(source_dump, 0, x);
  }

  public void visitArrayInit(/*@ non_null */ ArrayInit x) {
    visitVarInit(x);
    setVisitString(source_dump.toString(), x.dotId, "ArrayInit");
  }

  public void visitExpr(/*@ non_null */ Expr x) {
    // grab the source text
    printer.print(source_dump, 0, x);
  }

  public void visitThisExpr(/*@ non_null */ ThisExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "ThisExpr");
  }

  public void visitLiteralExpr(/*@ non_null */ LiteralExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "LiteralExpr");
  }

  public void visitArrayRefExpr(/*@ non_null */ ArrayRefExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "ArrayRefExpr");
  }

  public void visitNewInstanceExpr(/*@ non_null */ NewInstanceExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "NewInstanceExpr");
  }

  public void visitNewArrayExpr(/*@ non_null */ NewArrayExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "NewArrayExpr");
  }

  public void visitCondExpr(/*@ non_null */ CondExpr x) {
    setVisitString("", x.dotId, "CondExpr(have not seen yet)");
  }

  public void visitInstanceOfExpr(/*@ non_null */ InstanceOfExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "InstanceOfExpr");
  }

  public void visitCastExpr(/*@ non_null */ CastExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "CastExpr");
  }

  public void visitBinaryExpr(/*@ non_null */ BinaryExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "BinaryExpr");
  }

  public void visitUnaryExpr(/*@ non_null */ UnaryExpr x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "UnaryExpr");
  }

  public void visitParenExpr(/*@ non_null */ ParenExpr x) {
    setVisitString("", x.dotId, "ParenExpr(have not seen yet)");
  }

  public void visitAmbiguousVariableAccess(/*@ non_null */ AmbiguousVariableAccess x) {
    setVisitString("", x.dotId, "AmbiguousVariableAccess(have not seen yet)");
  }

  public void visitVariableAccess(/*@ non_null */ VariableAccess x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "VariableAccess");
  }

  public void visitFieldAccess(/*@ non_null */ FieldAccess x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "FieldAccess");
  }

  public void visitAmbiguousMethodInvocation(/*@ non_null */ AmbiguousMethodInvocation x) {
    setVisitString("", x.dotId, "AmbigousMethodInvocation(have not seen yet)");
  }

  public void visitMethodInvocation(/*@ non_null */ MethodInvocation x) {
    visitExpr(x);
    setVisitString(source_dump.toString(), x.dotId, "MethodInvocation");
  }

  public void visitClassLiteral(/*@ non_null */ ClassLiteral x) {
    setVisitString("", x.dotId, "ClassLiteral(have not seen yet)");
  }

  public void visitObjectDesignator(/*@ non_null */ ObjectDesignator x) {
    // get the source text
    printer.print(source_dump, 0, x);
  }

  public void visitExprObjectDesignator(/*@ non_null */ ExprObjectDesignator x) {
    visitObjectDesignator(x);
    setVisitString(source_dump.toString(), x.dotId, "ExprObjectDesignator");
  }

  public void visitTypeObjectDesignator(/*@ non_null */ TypeObjectDesignator x) {
    visitObjectDesignator(x);
    setVisitString("", x.dotId, "TypeObjectDesignator");
  }

  public void visitSuperObjectDesignator(/*@ non_null */ SuperObjectDesignator x) {
    visitObjectDesignator(x);
    setVisitString(source_dump.toString(), x.dotId, "SuperObjectDesignator");
  }

  public void visitType(/*@ non_null */ Type x) {
    // get the source text and set the visit string
    printer.print(source_dump, x);
    setVisitString(source_dump.toString(), x.dotId, "Type");
  }

  public void visitErrorType(/*@ non_null */ ErrorType x) {
    setVisitString("", x.dotId, "ErrorType(have not seen yet)");
  }

  public void visitPrimitiveType(/*@ non_null */ PrimitiveType x) {
    // grab the source text as a Type node can appear in the tree
    printer.print(source_dump, x);
    setVisitString(source_dump.toString(), x.dotId, "PrimitiveType");
  }

  public void visitTypeName(/*@ non_null */ TypeName x) {
    // grab the source text as the generic Type can appear in the tree
    printer.print(source_dump, x);
    setVisitString(source_dump.toString(), x.dotId, "TypeName");
  }

  public void visitArrayType(/*@ non_null */ ArrayType x) {
    // grab the source text as the generic Type can be in the tree as well
    printer.print(source_dump, x);
    setVisitString(source_dump.toString(), x.dotId, "ArrayType");
  }

  public void visitName(/*@ non_null */ Name x) {
    // grab the source text
    printer.print(source_dump, x);
  }

  public void visitSimpleName(/*@ non_null */ SimpleName x) {
    visitName(x);
    setVisitString(source_dump.toString(), x.dotId, "SimpleName");
  }

  public void visitCompoundName(/*@ non_null */ CompoundName x) {
    visitName(x);
    setVisitString(source_dump.toString(), x.dotId, "CompoundName");
  }

  public void visitModifierPragma(/*@ non_null */ ModifierPragma x) {
    setVisitString("", x.dotId, "ModifierPragma(have not seen yet)");
  }

  public void visitLexicalPragma(/*@ non_null */ LexicalPragma x) {
    setVisitString("", x.dotId, "LexicalPragma(have not seen yet)");
  }

  public void visitTypeModifierPragma(/*@ non_null */ TypeModifierPragma x) {
    setVisitString("", x.dotId, "TypeModifierPragma(have not seen yet)");
  }
  
}
