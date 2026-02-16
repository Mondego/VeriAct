package escjava.dfa.buildcfd;

import java.io.IOException;
import java.io.Writer;

import javafe.ast.Expr;
import javafe.util.Assert;
import javafe.ast.GenericVarDeclVec;
import escjava.ast.LoopCmd;
import escjava.dfa.cfd.CFD;
import escjava.dfa.cfd.CodeNode;
import escjava.dfa.cfd.Node;
import escjava.dfa.cfd.NodeVisitor;

public class NestedLoopCFDBuilder extends GCtoCFDBuilder {    
    protected CFD constructLoopGraph(LoopCmd loopCommand, GenericVarDeclVec scope) {
        CFD retv = new CFD();
        retv.createSimpleCFD(new CodeNode(scope, loopCommand));
        return retv;
    }
}
