package escjava.dfa.cfd;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.Writer;

import javafe.ast.Expr;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.PrettyPrint;
import javafe.util.Assert;
import escjava.ast.*;
import escjava.sp.SPVC;
import escjava.translate.GC;

/**
 * This class represents a node in control flow diagram that represent
 * a piece of code in program.
 */
public class CodeNode extends Node {

    private final /*@ non_null @*/ GuardedCmd code;

    /**
     * Initializes a new instance of the Node class.
     * @param scope  vector variables representing scope of the new node
     */
    public CodeNode(/*@ non_null @*/GenericVarDeclVec scope,
                    /*@ non_null @*/GuardedCmd code) {
        super(scope);
        this.code = code;
    }
    
    //@ private behavior
    //@    ensures \result == code;
    /*@ pure @*/public /*@ non_null @*/GuardedCmd getCode() {
        return code;
    }
    
    public void accept(/*@ non_null @*/NodeVisitor nodeVisitor) {
        nodeVisitor.visitCodeNode(this);
    }
   
    /**
     * Computes the stronest postcondition of this node, given a precondition.    
     * This implementation assumes that the code associated to this node is one of the commands ASSERT, ASSUME, SKIP.
     * @param pre precondition
     */
    public /*@ non_null @*/Expr computeSp(/*@ non_null @*/Expr pre) {
        int t = code.getTag();
        
        switch (t) {
        case TagConstants.ASSERTCMD:
        case TagConstants.ASSUMECMD:
            return GC.and(pre,((ExprCmd)code).pred);
        case TagConstants.SKIPCMD:
            return pre;
        default:
            Assert.fail("The command associated to this code is not supported by this method (" + this.getCode() + ").");
            return null;
        }
    }
    
    public /*@ non_null @*/String toString() {
        return "[code node: " + getCodeString() + "]";
     }
    
    //@ assignable \nothing;
    private String getCodeString() {
        String r = null;
        if (PrettyPrint.inst != null && PrettyPrint.inst instanceof EscPrettyPrint) {
            ByteArrayOutputStream bs = new ByteArrayOutputStream();
            ((EscPrettyPrint) PrettyPrint.inst).print(bs, 0, this.code);
            r = bs.toString();

        } else
            return r = code.toString();
        return r.replace('"', ' ').replace('|', ' ');
    }
    
     void printToDot(/*@ non_null @*/Writer dot, boolean bold) throws IOException {
         dot.write("" + hashCode() + " [label=\"" + getCodeString() + "\"" +
               (bold?",style=bold":"")  +"];\n");      
    }


    public Node shallowCopy() {
        Node clone;
        clone = new CodeNode(this.scope.copy(), this.code);
        return clone;
    }



}
