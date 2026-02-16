package escjava.dfa.cfd;

import java.io.IOException;
import java.io.Writer;

import javafe.ast.Expr;
import javafe.ast.GenericVarDeclVec;

/**
 * This class represents a coupling node in control flow diagram.
 */
public class CouplingNode extends Node {

    
    public CouplingNode(GenericVarDeclVec scope) {
        super(scope);
    }

    public void accept(/*@ non_null @*/NodeVisitor nodeVisitor) {
        nodeVisitor.visitCouplingNode(this);
    }

    public /*@ non_null @*/ Expr computeSp(/*@ non_null @*/Expr pre) {
        return (Expr)pre.clone();
    }
    
    public/*@ non_null */String toString() {
        return "[coupling node]";
    }

    void printToDot(/*@ non_null @*/Writer dot, boolean bold) throws IOException {
        dot.write("" + hashCode() + "[shape=point, width=0.15];\n");
    }

    public Node shallowCopy() {
        return new CouplingNode(this.scope.copy());
    }

   
}


