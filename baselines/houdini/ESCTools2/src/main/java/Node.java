package escjava.dfa.cfd;

import java.io.IOException;
import java.io.Writer;

import javafe.ast.Expr;
import javafe.ast.GenericVarDeclVec;
import escjava.ast.*;

/**
 * This class represents a node in control flow diagram.
 */
public abstract class Node {
    /** A variable scope of this node.  */
    GenericVarDeclVec scope;

    /** A node list of parents of this node.   */
    /*@ non_null @*/ NodeList parents;

    /** A node list of children of this node.   */
    /*@ non_null @*/ NodeList children;

    /**
     * Initializes a new instance of the Node class.
     *
     * @param scope  vector variables representing scope of the new node
     */
    public Node(GenericVarDeclVec scope) {
        this.scope = scope;
        this.parents = new NodeList();
        this.children = new NodeList();
    }
    
    /**
     * Creates a shallow copy of the node, the parents and children should remain emtpy. 
     * This method is used in cloning of the graph which takes care of the edges.
     */
    public abstract Node shallowCopy();
    


    /**
     * Returns the children node list.
     */
    /*@ pure */ public /*@ non_null */ NodeList getChildren() {
        return this.children;
    }

    /**
     * Returns parent node list.
     */
    /*@ pure */ public /*@ non_null */ NodeList getParents() {
        return this.parents;
    }

    /**
     * Returns variable  scope of this node.
     * @return variable scope for this node represented as a variable
     *         declaration vector
     */

    /*@ pure @*/ public GenericVarDeclVec  getScope() {
        return this.scope;
    }


    /**
     * Connects this node to the given node.
     *
     * @param to  to which node this node should be connected
     */
    public void connectTo(/*@ non_null @*/Node to) {
        this.children.addNode(to);
        to.parents.addNode(this);
    }
	
    public abstract void accept(/*@ non_null */NodeVisitor visitor);
 
    /**
     * Compute the strongest postcondition given a precondition.
     * @param pre the precondition
     */
    abstract public /*@ non_null */Expr computeSp(/*@ non_null */Expr pre); 
    
    abstract void printToDot(/*@ non_null */Writer dot, boolean bold) throws IOException;
}
