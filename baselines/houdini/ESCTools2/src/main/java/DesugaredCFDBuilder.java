package escjava.dfa.buildcfd;

import javafe.ast.GenericVarDeclVec;
import escjava.ast.LoopCmd;
import escjava.dfa.cfd.CFD;

public class DesugaredCFDBuilder extends GCtoCFDBuilder {

    protected CFD constructLoopGraph(LoopCmd loopCommand, GenericVarDeclVec scope) {
        return constructGraph(loopCommand.desugared, scope);
    }
}
