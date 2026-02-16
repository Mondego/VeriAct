package escjava.dfa.cfd;

import java.io.IOException;
import java.io.Writer;

import javafe.ast.Expr;
import javafe.ast.GenericVarDeclVec;

/**
 * This class represents an exception node in control flow diagram.
 */
public class ExceptionNode extends Node {
    private static final String strRepresentation = "[exception node]"; 
    
    public ExceptionNode () {
        super(GenericVarDeclVec.make(0));
    }

    public  void accept(/*@ non_null @*/NodeVisitor nodeVisitor) {
        nodeVisitor.visitExceptionNode(this);
    }
    
    public /*@ non_null @*/Expr computeSp(/*@ non_null @*/Expr pre) {
        return (Expr)pre.clone();
    }

    public /*@ non_null @*/String toString() {
        return strRepresentation;
    }

    void printToDot(/*@ non_null @*/Writer dot, boolean bold) throws IOException {
            dot.write("" + hashCode() + " [label=\"" + "RAISE" + "\"];\n");           
    }

    public Node shallowCopy() {
        return new ExceptionNode();
    }

}
