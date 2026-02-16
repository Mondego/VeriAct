package escjava.ast;

import javafe.ast.ASTNode;

/**
 * @title       "Abstract Syntax Tree Visitor"
 * @description "An abstract class for AST visitors"
 * @author      "Dermot Cochran"
 * @copyright   "(c) 2007 EU Mobius FP-6 IST-15909 project"
 * @license     "MIT open source"
 * @version     "$Revision: 1.1 $"
 */

/**
 * <p> An abstract subclass of the Visitor class which specialises in visiting
 * the nodes of an Abstract Syntax Tree.
 * 
 * @version 1.0 $Date: 2007/07/03 16:35:22 $
 * @author Dermot Cochran
 */

public abstract class ASTVisitor extends Visitor {
   
  public ASTVisitor() {
    super();
  }

  public void visitASTNode(/*@non_null*/ASTNode x) {
    // a child of this node
    Object child = null;
    // temporary ASTNode
    ASTNode temp = null;
    // visit each child in this ASTNode if the child is an ASTNode
    for(int count = 0; count < x.childCount(); count++) {
      child = x.childAt(count);
      if(child instanceof ASTNode) {
        temp = (ASTNode) child;
        temp.accept(this);
      }
    }
  }
}
